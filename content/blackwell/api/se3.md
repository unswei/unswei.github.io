---
title: "SE(3) state space"
description: "Rigid 3D transforms, xyzw quaternions, right body tangents and covariance transport in Blackwell 0.0.2."
weight: 25
lastmod: 2026-09-20T00:00:00+10:00
draft: false
aliases:
  - "/blackwell/reference/spaces/se3/"
---

Available from **0.0.2**. Import with `from blackwell.spaces import se3`.

| Quantity | Encoding |
| --- | --- |
| Pose | `[x, y, z, qx, qy, qz, qw]`, shape `(7,)` |
| Quaternion | Unit Hamilton quaternion, scalar-last `xyzw` |
| Tangent | `[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]`, shape `(6,)` |
| Gaussian covariance | `(6, 6)`, right/body tangent coordinates at the mean |
| Pose particles | `(N, 7)` states and `(N,)` normalised weights |
| Identity | `[0, 0, 0, 0, 0, 0, 1]` |

A pose \(T_{AB}\) maps coordinates in frame B into frame A:
\(p_A = R(q)p_B + t\). Rotations are active and right-handed.
`compose(T_AB, T_BC)` gives `T_AC`; the right argument acts first.
Translation and points use the application's length unit; rotation vectors
use radians. The tangent translation \(\rho\) is mapped by the exponential's
\(V(\phi)\) matrix and is generally different from the stored translation \(t\).

Retraction is `T Exp(delta)`. Thus `local_coordinates(reference, target)`
means `Log(reference^-1 target)`, and covariance belongs to the reference's
body tangent chart. To show why this matters: after a 90° rotation about z,
retracting by `[1, 0, 0, 0, 0, 0]` moves along parent-frame y, not x.

Quaternions are normalised on input to each operation; finite non-zero
quaternions are required. Both signs represent the same rotation. Group
operations preserve sign continuity rather than enforcing a preferred sign.
Use `local_coordinates` or point actions to compare transforms, rather than
subtracting their seven stored coordinates.

`log` chooses a rotation with norm at most pi, consistently for `q` and `-q`.
At an exact half-turn (`qw == 0`), its largest-magnitude vector component is
made non-negative, with the first component winning a tie. The logarithm is
discontinuous at this boundary. It cannot recover arbitrary rotation vectors
outside the principal ball, and its Jacobians must not be used on the cut.
Small-angle expansions support differentiation at the identity in both
forward and reverse mode.

`adjoint(T)` changes right/body perturbations to left/parent perturbations.
`transport(reference, target, P)` instead differentiates the change of local
chart about a new mean. These operations have different meanings.

All operations accept one pose or point at a time. Use `jax.vmap` to batch,
`jax.jit` to compile and `jax.lax.scan` to compose a trajectory. The kernels
preserve float32 or float64 inputs; enable JAX double precision when needed.

See [SE(3) uncertainty propagation](/blackwell/examples/se3-uncertainty/) for
composition, inversion, point Jacobians, correlated inputs and belief examples.

## Public operations

```python
compose(left: Array, right: Array) -> Array
inverse(pose: Array) -> Array
transform_point(pose: Array, point: Array) -> Array
exp(tangent: Array) -> Array
log(pose: Array) -> Array
retract(pose: Array, tangent: Array) -> Array
local_coordinates(reference: Array, pose: Array) -> Array
adjoint(pose: Array) -> Array
transport(reference: Array, target: Array, covariance: Array) -> Array
```

| Operation | Result and meaning |
| --- | --- |
| `compose(left, right)` | `(7,)` pose, with the right transform acting first |
| `inverse(pose)` | `(7,)` inverse frame map with conjugate quaternion |
| `transform_point(pose, point)` | `(3,)` parent-frame point; the input point has shape `(3,)` |
| `exp(tangent)` | `(7,)` pose from a `(6,)` body tangent; includes translation/rotation coupling |
| `log(pose)` | `(6,)` principal tangent with rotation norm at most pi |
| `retract(pose, tangent)` | `(7,)` pose equal to `compose(pose, exp(tangent))` |
| `local_coordinates(reference, pose)` | `(6,)` tangent at the reference, equal to `log(compose(inverse(reference), pose))` |
| `adjoint(pose)` | `(6, 6)` matrix `[[R, skew(t) @ R], [0, R]]`, mapping right/body to left/parent perturbations |
| `transport(reference, target, covariance)` | Symmetric `(6, 6)` covariance re-expressed in the target chart |

Covariance transport uses the derivative at zero of
`local_coordinates(target, retract(reference, delta))`, followed by
`J @ covariance @ J.T` and symmetrisation. It is a first-order chart change,
valid away from a relative rotation of pi. The zero-error chart displacement
need not vanish: transporting covariance alone does not move the mean.

[View the 0.0.2 implementation and complete docstrings](https://github.com/unswei/blackwell/blob/v0.0.2/src/blackwell/spaces/se3.py).

