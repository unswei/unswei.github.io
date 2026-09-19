---
title: "Choose an estimator"
description: "Decide between Blackwell's extended Kalman and bootstrap particle filters."
weight: 30
lastmod: 2026-09-20T00:00:00+10:00
draft: false
---

Blackwell currently provides an extended Kalman filter (EKF) and a bootstrap particle filter. Both use the same state spaces and model families.

| Question | Extended Kalman filter | Particle filter |
| --- | --- | --- |
| Belief shape | One local Gaussian | Weighted state samples |
| Best fit | Approximately unimodal uncertainty | Skewed, multimodal or strongly nonlinear uncertainty |
| Compute | Matrix operations in tangent dimension | Scales with particle count and measurement cost |
| Random key | Not needed for inference | Required for propagation and resampling |
| Differentiable/JIT | Yes, subject to model operations | Yes; resampling is discrete |
| Current support | Linear and SE(2) range-bearing | SE(2) range-bearing and compatible custom models |

SE(3) geometry is available from 0.0.2. Both filters accept `se3` with compatible custom dynamics and observation families. See [SE(3) uncertainty propagation](/blackwell/examples/se3-uncertainty/) for correlated transforms and points.

## Start with the EKF when

- your posterior should remain close to one mode;
- the initial estimate is reasonably good;
- you need compact beliefs or fast repeated updates; and
- linearisation around the mean is meaningful.

## Start with particles when

- the initial pose has several plausible modes;
- geometry or observations produce non-Gaussian posteriors;
- recovery from a poor initial estimate matters more than compactness; or
- you want to inspect the posterior shape directly.

<div class="blackwell-docs-note">
<p><strong>Current scope.</strong> Blackwell does not yet provide built-in IMU models, an unscented Kalman filter, SLAM state augmentation, data association, smoothing or production sensor integration. It is strongest as a small JAX-native estimation core and research or teaching scaffold.</p>
</div>

Continue with the [EKF guide](/blackwell/guides/extended-kalman-filter/) or [particle-filter guide](/blackwell/guides/particle-filter/).
