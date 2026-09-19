---
title: "Examples"
description: "Runnable public-API workflows for linear, Gaussian and particle state estimation."
weight: 40
draft: false
---

Every example is a standalone script under `examples/`, runs without plotting dependencies and uses only Blackwell's public API.

| Example | What it demonstrates | Command |
| --- | --- | --- |
| [Quick start](/blackwell/getting-started/quickstart/) | One compiled SE(2) EKF step | `uv run python examples/quickstart.py` |
| [Linear Kalman filter](/blackwell/examples/linear-kalman-filter/) | Euclidean model and `jax.lax.scan` | `uv run python examples/linear_kalman_filter.py` |
| [SE(2) EKF localisation](/blackwell/examples/se2-localisation/) | Simulation, manifold EKF, RMSE and NEES | `uv run python examples/se2_localisation.py` |
| [SE(2) particle localisation](/blackwell/examples/particle-localisation/) | Weighted particles, ESS and resampling | `uv run python examples/particle_localisation.py` |
| [SE(3) uncertainty](/blackwell/examples/se3-uncertainty/) | Correlated composition, inversion and transformed points | `uv run python examples/se3_uncertainty.py` |

Clone the repository and install all extras to reproduce figures:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
```

The examples fix their random seeds. Identical software and hardware should reproduce their trajectories, although small floating-point differences can occur across JAX backends.
