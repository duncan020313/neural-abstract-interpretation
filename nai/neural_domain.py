from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, env_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, env_vec: torch.Tensor) -> torch.Tensor:
        return self.net(env_vec)


class Join(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class Transfer(nn.Module):
    def __init__(self, hidden_dim: int, stmt_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stmt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor, stmt_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, stmt_vec], dim=-1))


class Predicate(nn.Module):
    def __init__(self, hidden_dim: int, expr_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + expr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, expr_vec: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([state, expr_vec], dim=-1))
        return torch.sigmoid(logits).squeeze(-1)


@dataclass
class NeuralDomain:
    env_dim: int
    stmt_dim: int
    expr_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        self.encoder = Encoder(self.env_dim, self.hidden_dim)
        self.join = Join(self.hidden_dim)
        self.transfer = Transfer(self.hidden_dim, self.stmt_dim)
        self.predicate = Predicate(self.hidden_dim, self.expr_dim)

    def init_state(self, env_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_vec is None:
            return torch.zeros(self.hidden_dim)
        return self.encoder(env_vec)
