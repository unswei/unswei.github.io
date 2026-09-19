---
title: "State spaces"
description: "Euclidean and right-retraction SE(2)/SE(3) state-space operations."
weight: 20
lastmod: 2026-09-20T00:00:00+10:00
draft: false
---

State spaces provide `retract`, `local_coordinates` and `transport`, allowing filters to remain independent of a state representation.

## Euclidean vectors

Module: `blackwell.spaces.euclidean`

```python
retract(state: Array, tangent: Array) -> Array
local_coordinates(reference: Array, state: Array) -> Array
transport(reference: Array, target: Array, covariance: Array) -> Array
```

- `retract` returns `state + tangent`.
- `local_coordinates` returns `state - reference`.
- `transport` returns the covariance unchanged.

State and tangent shapes match; covariance has shape `(state_dim, state_dim)`.

## SE(2)

Module: `blackwell.spaces.se2`

Blackwell represents a pose as `[x, y, heading]` and a local tangent as `[forward, lateral, turn]`. Retraction is right/body-frame.

### Group operations

```python
compose(left: Array, right: Array) -> Array
inverse(pose: Array) -> Array
exp(tangent: Array) -> Array
log(pose: Array) -> Array
adjoint(pose: Array) -> Array
wrap_angle(angle: Array) -> Array
```

- `compose` applies `right` relative to `left` and wraps heading to `[-pi, pi]`.
- `inverse` returns the inverse pose.
- `exp` maps `[forward, lateral, turn]` to a pose, with a stable series near zero turn.
- `log` maps a pose to its principal body tangent.
- `adjoint` returns the `(3, 3)` body-tangent adjoint matrix.
- `wrap_angle` accepts a scalar or array of radians and preserves its shape.

### Local-coordinate operations

```python
retract(pose: Array, tangent: Array) -> Array
local_coordinates(reference: Array, pose: Array) -> Array
transport(reference: Array, target: Array, covariance: Array) -> Array
```

`retract(pose, tangent)` is `compose(pose, exp(tangent))`. `local_coordinates(reference, pose)` returns the body tangent which retracts the reference towards the target.

`transport` maps a `(3, 3)` covariance from the reference tangent coordinates to the target coordinates using the exact first-order Jacobian of re-expression, then symmetrises the result. It is valid away from the logarithm's unavoidable principal-angle branch cut.

[View the Euclidean](https://github.com/unswei/blackwell/blob/main/src/blackwell/spaces/euclidean.py) and [SE(2) implementations](https://github.com/unswei/blackwell/blob/main/src/blackwell/spaces/se2.py).

## SE(3)

Version 0.0.2 adds `blackwell.spaces.se3`. Poses have shape `(7,)` in
`[x, y, z, qx, qy, qz, qw]` order; body tangents have shape `(6,)`, with
translation before rotation. See the [complete SE(3) reference](/blackwell/api/se3/)
for conventions, group operations, point actions, logarithm limits and transport.
