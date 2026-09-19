---
title: "Beliefs"
description: "Immutable Gaussian and particle belief containers."
weight: 10
lastmod: 2026-09-20T00:00:00+10:00
draft: false
---

Module: `blackwell.beliefs`

## `GaussianBelief`

```python
class GaussianBelief(NamedTuple):
    mean: Array
    covariance: Array
```

A Gaussian belief with covariance in local tangent coordinates. The container deliberately does not retain a state-space object, keeping it an uncomplicated JAX PyTree.

| Field | Meaning |
| --- | --- |
| `mean` | State array in the accompanying state space. For SE(2), `[x, y, heading]` with shape `(3,)`. For SE(3), `[x, y, z, qx, qy, qz, qw]` with shape `(7,)`. |
| `covariance` | Symmetric local covariance with shape `(tangent_dim, tangent_dim)` at `mean`. For SE(2), axes are body-frame `[forward, lateral, turn]`. For SE(3), the `(6, 6)` covariance uses body translation `rho` followed by rotation `phi`. |

## `ParticleBelief`

```python
class ParticleBelief(NamedTuple):
    particles: Array
    weights: Array
```

A normalised weighted collection of particles.

| Field | Meaning |
| --- | --- |
| `particles` | State samples with shape `(particle_count, *state_shape)`. |
| `weights` | Non-negative, normalised weights with shape `(particle_count,)`. |

SE(3) pose particles have shape `(N, 7)`. Sample six-dimensional tangent noise and retract it, rather than adding noise to quaternion components. Separate belief containers retain marginal uncertainty only; cross-covariances belong in explicit application data.

The container does not enforce normalisation at construction. Filter operations assume finite weights that sum to one.

[View `beliefs.py` on GitHub](https://github.com/unswei/blackwell/blob/main/src/blackwell/beliefs.py).
