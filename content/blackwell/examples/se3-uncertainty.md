---
title: "SE(3) uncertainty propagation"
description: "Correlated transform composition, inversion and uncertain 3D points, checked with Monte Carlo."
weight: 40
lastmod: 2026-09-20T00:00:00+10:00
draft: false
---

Compose uncertain transforms, invert them, and transform uncertain 3D points:

```console
uv run python examples/se3_uncertainty.py
```

This example needs Blackwell 0.0.2 or newer. It uses only public
operations and prints a reproducible Monte Carlo comparison without plotting
dependencies. Read the [encoding and frame conventions](/blackwell/api/se3/)
before adapting it to a sensor or calibration problem.

## Linearise in tangent coordinates

Write an uncertain transform as \(T=\bar T\operatorname{Exp}(\delta)\), with
\(\delta=[\rho,\phi]\) and a \(6\times6\) covariance. Its seven stored coordinates
are not the covariance axes. For a transform-valued function \(f\), differentiate
the local output error at \(\bar Y=f(\bar T)\):

```python
jacobian = jax.jacfwd(
    lambda delta: se3.local_coordinates(
        output_mean, function(se3.retract(input_mean, delta))
    )
)(jnp.zeros(6, dtype=input_mean.dtype))
```

`jax.jacrev` also works away from the principal-logarithm cut. Differentiating
the raw quaternion components produces a different, redundant coordinate
Jacobian and cannot directly propagate a six-dimensional pose covariance.

## Composition and inversion

Let \(C=AB\). For right perturbations about \(\bar A,\bar B,\bar C\),

$$
J_A=\operatorname{Ad}_{\bar B^{-1}},\qquad J_B=I_6.
$$

With \(P_{AB}=\operatorname{Cov}(\delta_A,\delta_B)\), the first-order result is

$$
P_C=J_A P_A J_A^T+P_B+J_A P_{AB}+P_{AB}^T J_A^T.
$$

The cross-covariance is between the respective **body tangent coordinates**
of A and B; shared map, calibration or odometry errors can make it non-zero.
Two `GaussianBelief` objects store only marginals, so they cannot encode or
reconstruct this cross-covariance. Supply it from the joint model. Zero is an
explicit independence assumption.

For inversion \(Y=A^{-1}\), the right-tangent Jacobian is
\(J_A=-\operatorname{Ad}_{\bar A}\) and \(P_Y=J_A P_A J_A^T\). If Y is used
together with A, retain their cross-covariance \(P_{AY}=P_AJ_A^T\). For example,
\(AA^{-1}\) has zero uncertainty; treating those inputs as independent loses
this cancellation. A regression test checks this singular joint distribution.

The script constructs the Jacobians using `retract`, `local_coordinates` and
`jax.jacfwd`; no operation-specific Jacobian helper is needed.

## Transforming correlated uncertain points

For \(y=Tp=Rp+t\), let the point perturbation be additive in T's input frame.
At the nominal pose and point,

$$
J_T=\begin{bmatrix}R&-R[p]_\times\end{bmatrix},\qquad J_p=R.
$$

Let \(P_{Tp}=\operatorname{Cov}(\delta_T,\delta_p)\) have shape \((6,3)\).
Propagate the full joint covariance:

$$
P_y=
\begin{bmatrix}J_T&J_p\end{bmatrix}
\begin{bmatrix}P_T&P_{Tp}\\P_{Tp}^T&P_p\end{bmatrix}
\begin{bmatrix}J_T&J_p\end{bmatrix}^T.
$$

Equivalently, this is \(J_TP_TJ_T^T+J_pP_pJ_p^T\) plus the two cross terms
\(J_TP_{Tp}J_p^T+J_pP_{Tp}^TJ_T^T\). The joint matrix must be symmetric
positive semidefinite; arbitrary cross blocks need not define a valid model.
The example builds it from a shared latent factor, so this condition holds.

The Monte Carlo calculation draws **paired errors from that same joint
distribution**, retracts each pose error, adds each point error and transforms
each paired sample. Sampling the marginals independently would silently remove
the correlation being tested.

The script compares the first-order covariance with 50,000 transformed samples
and separately reports the error caused by discarding the cross-covariance.
This tests the small-noise approximation. For large errors or rotations near
the logarithm's cut, use the particle representation to inspect the resulting
distribution. The nominal transformed point need not equal its nonlinear
expectation; a small second-order mean shift is expected.

## Belief contracts and filters

- A pose `GaussianBelief` has a `(7,)` mean and `(6, 6)` local covariance.
- A pose `ParticleBelief` has `(N, 7)` states and `(N,)` weights. Draw six
  tangent errors and retract them; do not add noise to quaternion components.
- A transformed point is Euclidean: its Gaussian mean/covariance are `(3,)`
  and `(3, 3)`, and its particle states are `(N, 3)`.
- Neither container retains cross-covariance with another belief. Keep joint
  uncertainty in application data and pass it explicitly when propagating.

The existing `ExtendedKalmanFilter` and `BootstrapParticleFilter` accept
`se3` as their state-space module when supplied with compatible dynamics and
observation operations. Integration tests exercise prediction, correction,
particle initialisation/resampling, `jit`, `vmap`, `scan` and EKF gradients.
This release adds geometry; application-specific 3D sensors and IMU models
remain custom models.

## Reference result

On the documented CPU configuration, the seeded 50,000-sample example gives a
**0.69% relative covariance error**. Discarding the cross-covariance increases
the error to approximately **29%**. Small numerical differences between JAX
versions and devices are expected.

[Download the release example](https://github.com/unswei/blackwell/blob/v0.0.2/examples/se3_uncertainty.py).

## Complete example

```python
"""Correlated SE(3) uncertainty propagation using only Blackwell's public API.

Run with ``uv run python examples/se3_uncertainty.py``. Pose covariances use
right/body tangents [rho, phi]; point covariances use Cartesian coordinates.
Cross-covariances are supplied explicitly, never inferred from two marginals.
"""

import jax
import jax.numpy as jnp
from jax import Array

from blackwell import GaussianBelief, ParticleBelief
from blackwell.spaces import se3


def compose_beliefs(
    left: GaussianBelief, right: GaussianBelief, cross_covariance: Array
) -> GaussianBelief:
    """Compose poses with Cov(delta_left, delta_right), shape (6, 6)."""
    mean = se3.compose(left.mean, right.mean)

    def residual(delta):
        value = se3.compose(
            se3.retract(left.mean, delta[:6]),
            se3.retract(right.mean, delta[6:]),
        )
        return se3.local_coordinates(mean, value)

    jacobian = jax.jacfwd(residual)(jnp.zeros(12, dtype=mean.dtype))
    joint = jnp.block(
        [
            [left.covariance, cross_covariance],
            [cross_covariance.T, right.covariance],
        ]
    )
    covariance = jacobian @ joint @ jacobian.T
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def invert_belief(belief: GaussianBelief) -> GaussianBelief:
    """Invert a pose, expressing output covariance at the inverse mean."""
    mean = se3.inverse(belief.mean)
    jacobian = jax.jacfwd(
        lambda delta: se3.local_coordinates(
            mean, se3.inverse(se3.retract(belief.mean, delta))
        )
    )(jnp.zeros(6, dtype=mean.dtype))
    covariance = jacobian @ belief.covariance @ jacobian.T
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def transform_point_beliefs(
    pose: GaussianBelief, point: GaussianBelief, cross_covariance: Array
) -> GaussianBelief:
    """Transform a point with Cov(delta_pose, delta_point), shape (6, 3)."""
    mean = se3.transform_point(pose.mean, point.mean)

    def transform(delta):
        return se3.transform_point(
            se3.retract(pose.mean, delta[:6]), point.mean + delta[6:]
        )

    jacobian = jax.jacfwd(transform)(jnp.zeros(9, dtype=mean.dtype))
    joint = jnp.block(
        [
            [pose.covariance, cross_covariance],
            [cross_covariance.T, point.covariance],
        ]
    )
    covariance = jacobian @ joint @ jacobian.T
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def point_scenario():
    """Small correlated pose/point errors from a positive-definite joint model."""
    scales = jnp.array([0.015, 0.02, 0.01, 0.006, 0.008, 0.007, 0.02, 0.015, 0.025])
    factor = jnp.eye(9).at[6, 0].set(0.8).at[7, 5].set(0.7).at[8, 4].set(-0.6)
    factor = scales[:, None] * factor
    joint = factor @ factor.T
    pose = GaussianBelief(
        se3.exp(jnp.array([1.0, -0.4, 0.3, 0.4, -0.2, 0.5])), joint[:6, :6]
    )
    point = GaussianBelief(jnp.array([2.0, -1.0, 0.5]), joint[6:, 6:])
    return pose, point, joint


def sample_transformed_points(
    key: Array,
    pose: GaussianBelief,
    point: GaussianBelief,
    joint_covariance: Array,
    sample_count: int,
) -> ParticleBelief:
    """Draw paired errors jointly, preserving pose/point correlation."""
    errors = jax.random.normal(key, (sample_count, 9), dtype=pose.mean.dtype)
    errors = errors @ jnp.linalg.cholesky(joint_covariance).T
    weights = jnp.full(sample_count, 1 / sample_count, dtype=pose.mean.dtype)
    poses = ParticleBelief(
        jax.vmap(se3.retract, in_axes=(None, 0))(pose.mean, errors[:, :6]), weights
    )
    points = point.mean + errors[:, 6:]
    return ParticleBelief(
        jax.vmap(se3.transform_point)(poses.particles, points), poses.weights
    )


def empirical_gaussian(particles: ParticleBelief) -> GaussianBelief:
    """Summarise the Euclidean transformed-point cloud (population covariance)."""
    mean = particles.weights @ particles.particles
    centred = particles.particles - mean
    covariance = centred.T @ (particles.weights[:, None] * centred)
    return GaussianBelief(mean, covariance)


def main() -> None:
    pose, point, joint = point_scenario()
    # A shared latent source correlates both transform inputs.
    left_factor = jnp.diag(jnp.array([0.02, 0.01, 0.03, 0.006, 0.008, 0.01]))
    right_factor = left_factor * 0.5
    left = GaussianBelief(pose.mean, left_factor @ left_factor.T)
    right = GaussianBelief(
        se3.exp(jnp.array([0.2, 0.1, -0.3, -0.1, 0.2, 0.1])),
        right_factor @ right_factor.T + jnp.eye(6) * 1e-5,
    )
    composed = jax.jit(compose_beliefs)(left, right, left_factor @ right_factor.T)
    inverted = jax.jit(invert_belief)(composed)
    print("Composed pose [x, y, z, qx, qy, qz, qw]:", composed.mean)
    print("Composed body covariance diagonal:", jnp.diag(composed.covariance))
    print("Inverse body covariance diagonal:", jnp.diag(inverted.covariance))

    predicted = jax.jit(transform_point_beliefs)(pose, point, joint[:6, 6:])
    independent = transform_point_beliefs(pose, point, jnp.zeros((6, 3)))
    cloud = jax.jit(sample_transformed_points, static_argnames="sample_count")(
        jax.random.key(2026), pose, point, joint, sample_count=50_000
    )
    empirical = empirical_gaussian(cloud)

    def relative_error(covariance):
        return jnp.linalg.norm(covariance - empirical.covariance) / jnp.linalg.norm(
            empirical.covariance
        )

    print("Transformed point:", predicted.mean)
    print("Monte Carlo mean:", empirical.mean)
    print("First-order covariance:\n", predicted.covariance)
    print("Monte Carlo covariance:\n", empirical.covariance)
    print(
        f"Relative covariance error: {float(relative_error(predicted.covariance)):.2%}"
    )
    print(
        "Error if correlation is discarded:",
        f"{float(relative_error(independent.covariance)):.2%}",
    )


if __name__ == "__main__":
    main()
```

The geometry and tangent-linearisation background is described in
[Solà, Deray and Atchuthan, A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537)
and [Eade, Lie Groups for 2D and 3D Transformations](https://ethaneade.com/lie.pdf).
The formulas above use Blackwell's right/body perturbation convention.
