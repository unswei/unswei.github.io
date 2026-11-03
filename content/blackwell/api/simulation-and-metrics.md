---
title: "Simulation and metrics"
description: "Trajectory simulation, RMSE, consistency and finite-candidate decision metrics."
weight: 50
draft: false
---

## `Simulator`

Module: `blackwell.simulation`

```python
@dataclass(frozen=True)
class Simulator:
    state_space: StateSpaceOperations
    dynamics: DynamicsOperations
    observation: ObservationOperations
```

The simulator propagates a state, applies local process noise, then samples additive measurement noise in flattened measurement coordinates.

### `step`

```python
step(
    key: Array,
    state: Array,
    dynamics_model: object,
    observation_model: object,
    control: Array,
) -> tuple[Array, Array]
```

Returns `(next_state, measurement)`. The key is split once for process and measurement noise.

### `rollout`

```python
rollout(
    key: Array,
    initial_state: Array,
    dynamics_model: object,
    observation_model: object,
    controls: Array,
) -> tuple[Array, Array]
```

Uses `jax.lax.scan` to return `(states, measurements)` with one entry per control. The initial state itself is not included in `states`.

## Metrics

Module: `blackwell.metrics`

### Root mean square error

```python
root_mean_square_error(
    errors: Array,
    axis: int | tuple[int, ...] = 0,
) -> Array
```

Returns component-wise RMSE, preserving axes that are not reduced.

### Planar position RMSE

```python
position_rmse(
    errors: Array,
    axis: int | tuple[int, ...] = 0,
) -> Array
```

Expects tangent errors with planar position in the first two coordinates and returns root mean squared Euclidean position norm.

### NEES

```python
normalised_estimation_error_squared(
    errors: Array,
    covariances: Array,
) -> Array

mean_nees(
    errors: Array,
    covariances: Array,
    axis: int | tuple[int, ...] = 0,
) -> Array
```

`errors` has shape `(..., tangent_dim)` and matching nonsingular covariances have shape `(..., tangent_dim, tangent_dim)`. NEES output has shape `errors.shape[:-1]`; `mean_nees` averages it over selected sample axes.

## Decision metrics

Added in 0.0.4 in `blackwell.metrics`. Each call handles one fixed-capacity
candidate bank with shape `(K,)`, `K > 0`; use `vmap` for independent roots and
explicit `jit` for compilation. Labels and masks are Boolean arrays.

```python
selection_regret(
    reference_costs: Array,
    reference_accepted: Array,
    selected_index: Array,
    *,
    candidate_mask: Array,
    cost_observed: Array,
    constraint_observed: Array,
) -> SelectionRegret

binary_decision_counts(
    predicted_accepted: Array,
    reference_accepted: Array,
    *,
    candidate_mask: Array,
    constraint_observed: Array,
) -> BinaryDecisionCounts
```

`selection_regret` consumes the caller's scalar integer index, including `-1`
for no selection. It compares the selected reference cost with the best
occupied candidate whose cost is observed and finite and whose observed
constraint label accepts. Ties use stable candidate order. Out-of-range,
negative and padded indices are unavailable and cannot wrap into valid slots.

`SelectionRegret` is an immutable PyTree of scalar arrays:

| Field | Meaning |
| --- | --- |
| `regret` | Selected cost minus best feasible reference cost |
| `best_reference_cost`, `best_reference_index` | Best observed feasible candidate in this bank |
| `reference_available` | Reference set is nonempty |
| `selection_available` | Supplied index is in range and occupied |
| `selected_cost_available` | Selected cost is observed and finite |
| `selected_constraint_observed` | Selected reference constraint label is known |
| `selected_feasible` | That observed label accepts |
| `regret_defined` | Selection belongs to the feasible reference set |

Undefined costs and regret use zero placeholders, and an unavailable best
index uses `-1`; retain their flags. A false `selected_feasible` with a false
`selected_constraint_observed` does not establish infeasibility.

`BinaryDecisionCounts` retains integer `correct_acceptance`, `false_acceptance`,
`false_rejection` and `correct_rejection` counts over observed occupied slots.
It also returns `compared_count`, `unobserved_count`, `excluded_count`,
`model_accepted_count` and `reference_accepted_count`. Exclusions include padding
and missing truth; unobserved counts include only occupied slots.

`false_acceptance_rate` divides false accepts by observed model accepts;
`false_rejection_rate` divides false rejects by observed reference accepts.
Their corresponding `*_rate_defined` flags are false at zero denominators,
with zero placeholders for those rates. Sum counts before deriving pooled
rates. Candidates within a root are not independent statistical trials.

The [decision evaluation guide](/blackwell/guides/decision-relevant-evaluation/)
explains matched criteria, observation masks, completion windows, finite-bank
regret and paired executions. The helpers consume arrays and do not own a
selector or simulator. Task definitions and timing extraction stay in the
example; these metrics do not establish probabilistic safety.

[View simulation](https://github.com/unswei/blackwell/blob/main/src/blackwell/simulation.py) and [metrics](https://github.com/unswei/blackwell/blob/main/src/blackwell/metrics.py) source.
