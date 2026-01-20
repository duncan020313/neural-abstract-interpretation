from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import nn

from .abstract import encode_expr, encode_stmt
from .cfg import CFG
from .concrete import ConcreteResult, Value, eval_expr
from .neural_domain import NeuralDomain


@dataclass(frozen=True)
class TrainingStats:
    total_loss: float
    transfer_loss: float
    predicate_loss: float
    steps: int


def write_loss_plot(*, stats: List[TrainingStats], path: Union[str, Path]) -> None:
    """Write a training loss plot using matplotlib.

    The output format is inferred from the filename (e.g. .png, .svg, .pdf).
    """
    if not stats:
        return

    out_path = Path(path)
    epochs = list(range(len(stats)))
    total = [s.total_loss for s in stats]
    transfer = [s.transfer_loss for s in stats]
    predicate = [s.predicate_loss for s in stats]

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for write_loss_plot()."
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, total, label="total", linewidth=2)
    ax.plot(epochs, transfer, label="transfer", linewidth=2)
    ax.plot(epochs, predicate, label="predicate", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def train_domain_on_concrete_traces(
    *,
    cfg: CFG,
    domain: NeuralDomain,
    variables: List[str],
    runs: List[ConcreteResult],
    epochs: int = 10,
    lr: float = 1e-3,
) -> List[TrainingStats]:
    if epochs <= 0:
        return []
    if lr <= 0:
        raise ValueError("lr must be > 0.")
    if not runs:
        raise ValueError("Need at least one concrete run to train.")

    domain.encoder.train()
    domain.transfer.train()
    domain.predicate.train()
    domain.join.train()

    params = [
        *domain.encoder.parameters(),
        *domain.transfer.parameters(),
        *domain.predicate.parameters(),
        *domain.join.parameters(),
    ]
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    stats: List[TrainingStats] = []
    for _epoch in range(epochs):
        total_loss = torch.tensor(0.0)
        transfer_loss = torch.tensor(0.0)
        predicate_loss = torch.tensor(0.0)
        steps = 0

        for run in runs:
            trace = run.trace
            if len(trace) < 2:
                continue
            for i in range(len(trace) - 1):
                curr = trace[i]
                nxt = trace[i + 1]

                func_cfg = cfg.functions[curr.function]
                node = func_cfg.nodes[curr.pc]

                state = domain.encoder(_env_to_vec(curr.env, variables))
                target_state = domain.encoder(_env_to_vec(nxt.env, variables)).detach()

                if node.kind in {"branch", "loop_cond"}:
                    if node.expr is None:
                        continue
                    y = 1.0 if bool(eval_expr(node.expr, curr.env)) else 0.0
                    prob = domain.predicate(state, encode_expr(node.expr))
                    loss = bce(prob, torch.tensor(y))
                    predicate_loss = predicate_loss + loss
                    total_loss = total_loss + loss
                    steps += 1
                    continue

                stmt_vec = encode_stmt(node.stmt)
                pred_state = domain.transfer(state, stmt_vec)
                loss = mse(pred_state, target_state)
                transfer_loss = transfer_loss + loss
                total_loss = total_loss + loss
                steps += 1

        if steps == 0:
            stats.append(
                TrainingStats(
                    total_loss=0.0, transfer_loss=0.0, predicate_loss=0.0, steps=0
                )
            )
            continue

        opt.zero_grad()
        (total_loss / float(steps)).backward()
        opt.step()

        stats.append(
            TrainingStats(
                total_loss=float(total_loss.item() / float(steps)),
                transfer_loss=float(transfer_loss.item() / float(steps)),
                predicate_loss=float(predicate_loss.item() / float(steps)),
                steps=steps,
            )
        )
    return stats


def _env_to_vec(env: Dict[str, Value], variables: List[str]) -> torch.Tensor:
    values: List[float] = []
    for name in variables:
        v = env.get(name, 0)
        if isinstance(v, bool):
            values.append(1.0 if v else 0.0)
        else:
            values.append(float(v))
    return torch.tensor(values, dtype=torch.float32)

