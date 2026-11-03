---
title: "Simulation and evaluation"
description: "Generate reproducible trajectories and measure estimator accuracy and consistency."
weight: 30
draft: false
---

Blackwell exposes simulation and metrics alongside inference so an estimator can be checked on reproducible trajectories before it meets real sensor data.

## Simulate from the same models

`Simulator` uses the same state-space, dynamics and observation operations as the filters:

```python
import jax
import jax.numpy as jnp

from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.simulation import Simulator
from blackwell.spaces import se2

simulator = Simulator(se2, se2_models, range_bearing)
controls = jnp.tile(jnp.array([0.12, 0.0, 0.025]), (80, 1))
states, measurements = simulator.rollout(
    jax.random.key(1),
    initial_state=jnp.array([0.0, 0.0, 0.0]),
    dynamics_model=dynamics,
    observation_model=observation,
    controls=controls,
)
```

Each step propagates the deterministic model, samples local process noise, retracts it onto the state space and samples measurement noise. Passing the same key and inputs reproduces the same rollout.

## Compute errors in the right coordinates

Subtracting SE(2) pose arrays gives the wrong heading behaviour and ignores the body frame. Use the state space:

```python
errors = jax.vmap(se2.local_coordinates)(estimated_states, states)
```

The errors now share coordinates with the filter covariance.

## Report accuracy and consistency

```python
from blackwell.metrics import mean_nees, position_rmse

rmse = position_rmse(errors)
nees = mean_nees(errors, estimated_covariances)
```

Position RMSE answers “how far away was the estimate?” NEES answers “was that error plausible under the reported covariance?” For a calibrated Gaussian estimator, expected NEES is approximately the tangent dimension when averaged over enough independent trials.

One trajectory is useful for debugging, not statistical validation. For a meaningful consistency check:

1. split independent keys for many trials;
2. use `jax.vmap` to run them in parallel;
3. compute NEES at each time and across trials; and
4. compare against chi-squared confidence bounds for the sample count and tangent dimension.

## Evaluate decisions alongside prediction accuracy

Use the same initial states, tasks and candidate banks to check whether model
errors change selections, constraints or completion forecasts. The
[decision-relevant evaluation guide](/blackwell/guides/decision-relevant-evaluation/)
introduces finite-bank selection regret and observed binary decision counts,
then checks completion windows and paired open-loop executions in a planar
example. Cost, constraint and completion availability have separate masks.

## Move from simulation to sensors

Keep estimator code unchanged and replace only the source of controls and measurements. Confirm:

- timestamps and control integration intervals agree;
- headings and bearings are radians;
- landmarks and poses share a world frame;
- process covariance describes local post-motion uncertainty; and
- measurement ordering matches landmark ordering.

The [SE(2) example](/blackwell/examples/se2-localisation/) shows simulation, filtering and evaluation end to end.
