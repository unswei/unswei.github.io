---
title: "Blackwell"
description: "JAX-native probabilistic robotics: manifold-aware Gaussian and particle state estimation, simulation and consistency metrics."
date: 2026-08-27T00:00:00+10:00
lastmod: 2026-09-20T00:00:00+10:00
weight: 1
draft: false
showChildren: false
image: "/images/blackwell/og.png"
images:
  - "/images/blackwell/og.png"
---

Blackwell is a compact probabilistic-robotics library for manifold-aware Gaussian and particle inference—pure, immutable and designed for `jax.jit`, `jax.vmap` and `jax.lax.scan`.

<div class="project-actions">
  <a class="content-action" href="/blackwell/getting-started/installation/">Install Blackwell</a>
  <a class="content-action content-action-secondary" href="/blackwell/getting-started/quickstart/">Run the quick start</a>
  <a class="content-action content-action-secondary" href="/blackwell/api/">Browse the API</a>
</div>

<div class="blackwell-docs-facts">
  <div class="blackwell-docs-fact"><strong>SE(2) and SE(3)</strong><small>Covariance lives in local tangent coordinates.</small></div>
  <div class="blackwell-docs-fact"><strong>Transform-ready</strong><small>Pure kernels compose with JAX transformations.</small></div>
  <div class="blackwell-docs-fact"><strong>Two estimators</strong><small>Extended Kalman and bootstrap particle filters.</small></div>
  <div class="blackwell-docs-fact"><strong>Evaluation built in</strong><small>Simulation, RMSE and NEES are public APIs.</small></div>
</div>

Version **0.0.2** adds SE(3) transforms and correlated uncertainty propagation.
Read the [SE(3) example](/blackwell/examples/se3-uncertainty/) or the
[release notes](https://github.com/unswei/blackwell/releases/tag/v0.0.2).

## What can I do with it today?

<div class="blackwell-docs-cards">
  <a class="blackwell-docs-card" href="/blackwell/guides/extended-kalman-filter/"><strong>Localise a planar robot</strong><small>Estimate an SE(2) pose from body-frame motion and landmark range-bearing observations.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/examples/linear-kalman-filter/"><strong>Track linear systems</strong><small>Use the generic EKF as an ordinary Kalman filter with Euclidean linear models.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/guides/particle-filter/"><strong>Represent non-Gaussian beliefs</strong><small>Run a bootstrap particle filter with log-space weighting and explicit resampling.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/examples/se3-uncertainty/"><strong>Propagate 3D uncertainty</strong><small>Compose and invert SE(3) transforms, and transform correlated uncertain points.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/guides/simulation-and-evaluation/"><strong>Test consistency</strong><small>Generate reproducible trajectories and evaluate errors with RMSE and NEES.</small></a>
</div>

## A first estimate

Configure an estimator by pairing a state space with dynamics and observation families. Parameter values remain ordinary JAX PyTrees.

```python
import jax

from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
compiled_step = jax.jit(filter_.step)
```

The [five-minute localisation](/blackwell/getting-started/quickstart/) continues with a complete, runnable update.

## Design stance

- **JAX is the numerical backend.** Arrays, random keys and compilation remain explicit.
- **Geometry owns geometry.** State spaces implement retraction, local coordinates and covariance transport; filters stay generic.
- **Models own uncertainty.** Dynamics and observations describe noise and Jacobians; filters perform inference.
- **Shape changes are explicit.** Particle count, landmark count and trajectory length are static under JIT.

<div class="blackwell-docs-note">
<p><strong>Project maturity.</strong> Blackwell is useful today but remains pre-alpha. The supported surface is deliberately small, and API changes are possible before version 1.0. See <a href="/blackwell/getting-started/choose-an-estimator/">Choose an estimator</a> before selecting it for a long-lived deployment.</p>
</div>

## Documentation map

<div class="blackwell-docs-cards">
  <a class="blackwell-docs-card" href="/blackwell/getting-started/"><strong>Start here</strong><small>Install the package, run a first localisation and choose an estimator.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/concepts/"><strong>Concepts</strong><small>Understand beliefs, tangent coordinates, models, filters and JAX execution.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/guides/"><strong>User guides</strong><small>Build complete EKF, particle-filter, simulation and custom-model workflows.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/examples/"><strong>Examples</strong><small>Run linear, SE(2) and SE(3) uncertainty workflows with reproducible outputs.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/api/"><strong>API reference</strong><small>Look up public containers, operations, signatures, shapes and edge cases.</small></a>
  <a class="blackwell-docs-card" href="/blackwell/help/"><strong>Help</strong><small>Troubleshoot installation, JIT, numerical behaviour and platforms.</small></a>
</div>

Blackwell is open source under Apache-2.0. [Install it from PyPI](https://pypi.org/project/blackwell/) or [view the source code](https://github.com/unswei/blackwell).
