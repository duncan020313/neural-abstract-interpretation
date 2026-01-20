# Detailed Semantics: Neural Abstract Interpretation

## 1. Language Definition
### 1.1 Syntax (Simplified C with Functions)
Expressions:
- `n` integer literal
- `b` boolean literal
- `x` variable
- `e1 + e2`, `e1 - e2`, `e1 * e2`
- `e1 == e2`, `e1 < e2`, `e1 <= e2`
- `!e`, `e1 && e2`, `e1 || e2`

Statements:
- `x = e;`
- `if (e) { s* } else { s* }`
- `while (e) { s* }`
- `return e;`
- `f(e1, ..., ek);` (call as statement for side-effects)

Functions:
- `f(x1, ..., xk) { s* }`

Program:
- `P = { f1, ..., fn }` with a designated `main`.

### 1.2 Values and State
- Values: `v ::= int | bool`
- Environment: `env : Var -> v`
- Call stack: list of frames `(f, env, pc)` where `pc` is a CFG node.
- State: `sigma = (env, stack, pc)`

## 2. Concrete Semantics
We define a big-step evaluator for expressions and a small-step evaluator for statements.

### 2.1 Expression Evaluation
`eval(env, e) = v` defined in the usual way for arithmetic and boolean operators.

### 2.2 Statement Semantics (Small-Step)
Let `step(P, sigma) = sigma'` be a transition:
- Assignment:
  - If `pc` points to `x = e;` then
    - `env' = env[x <- eval(env, e)]`
    - advance to next `pc`.
- If:
  - If `pc` points to `if (e) { S1 } else { S2 }` then
    - evaluate `e` and jump to the entry `pc` of `S1` or `S2`.
- While:
  - If `pc` points to `while (e) { S }` then
    - evaluate `e` and jump to `S` or loop exit.
- Return:
  - If `pc` points to `return e;` then
    - evaluate `e`, pop stack, bind return to caller.
- Call:
  - If `pc` points to `f(args);` then
    - evaluate args, push new frame with callee env, jump to callee entry.

### 2.3 Trace Collection
Concrete execution produces a trace:
`tau = (pc0, env0), (pc1, env1), ...`
used for supervision.

## 3. Neural Abstract Domain
### 3.1 Abstract State
Abstract state is a vector `h in R^d`.
Concrete state abstraction:
`alpha(env, pc, ctx) = Enc([env], [pc], [ctx]) -> h`

Context `ctx` can include nearby statements or CFG neighborhood encodings.

### 3.2 Abstract Operators
Neural operators are parameterized functions:
- Join: `h1 ⊔ h2 = g_join(h1, h2, ctx)`
- Meet: `h1 ⊓ h2 = g_meet(h1, h2, ctx)`
- Transfer: `F_stmt(h) = g_stmt(h, enc(stmt), ctx)`

Monotonicity is encouraged by training or architectural constraints.

### 3.3 Abstract Concretization
`gamma(h)` is implicit via a decoder `Dec(h)` that predicts constraints,
or via the loss from traces without explicit decoding.

## 4. Abstract Semantics
We define abstract transitions over CFG nodes.

### 4.1 Abstract Transfer
For each statement `s`:
`h' = F_s(h)`

### 4.2 Abstract Join at Merge
At CFG merge nodes:
`h_merge = g_join(h1, h2, ctx)`

### 4.3 Soft Branching
Replace hard branching with a soft merge:
- Compute predicate score: `p = sigmoid(g_pred(h, enc(e)))`
- Then `h_if = p * h_then + (1 - p) * h_else`
This is differentiable and approximates hard branching.

### 4.4 Loop Fixpoint
Loop header state `h0` is updated iteratively:
```
h_{k+1} = g_join(h0, F_body(h_k), ctx)
```
Use damping for stability:
`h_{k+1} = (1 - beta) * h_k + beta * h_{k+1}`.
Stop after `K` steps or when `||h_{k+1} - h_k|| < eps`.

## 5. Context-Aware Operators
Operators accept a context vector:
`ctx = EncCtx(stmt_window, cfg_features, callstack_summary)`
This allows abstract operations to depend on surrounding program text.

## 6. Differentiable Analysis Objective
Given trace pairs `(pc, env)`:
- Encode concrete env to target vector `t = EncTarget(env)`.
- Predict analysis state `h(pc)` from abstract interpreter.
Loss components:
- Under-approx penalty: encourage `h` to include observed states.
- Over-approx penalty: discourage predicting states not in trace.
- Consistency penalty: encourage join/meet to be idempotent and commutative.

Example loss:
`L = L_under(h, t) + lambda_over * L_over(h, t) + lambda_cons * L_cons`

## 7. Soundness Notes
Exact soundness is not guaranteed; training aims to approximate an
over-approximating domain. If needed, use a conservative fallback:
- Blend with a traditional abstract domain at joins.
- Enforce monotone networks and widen iterations.

## 8. Outputs
The analysis returns:
- Abstract state per CFG node.
- Optionally, decoded invariants (intervals or predicate templates).
