---
title: "Development and contributing"
description: "Set up a checkout, run checks and understand Blackwell's supported public surface."
weight: 30
lastmod: 2026-11-03
draft: false
---

## Local environment

Blackwell uses `uv.lock` for a reproducible development environment:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
```

Run the core checks:

```console
uv run ruff check .
uv run pytest
uv run python -m build
uv run mkdocs build --strict
```

The repository's [contribution guide](https://github.com/unswei/blackwell/blob/main/CONTRIBUTING.md) records branch, testing and documentation expectations.

## Repository layout

```text
src/blackwell/
  beliefs.py           immutable Gaussian and particle containers
  spaces/              Euclidean, SE(2) and SE(3) geometry
  models/              dynamics and observation families
  filters/             EKF and bootstrap particle inference
  simulation.py        reproducible model rollouts
  metrics.py           error and consistency metrics
  _experiments/        private regression/reference implementations
examples/              runnable public-API workflows
tests/                 numerical and transformation tests
benchmarks/            explicit compilation and execution measurements
docs/                  source documentation and local reference build
```

## Public interface contract

- Numerical values are JAX arrays.
- Public data containers are immutable PyTrees.
- State spaces own retraction, local coordinates and covariance transport.
- Dynamics own propagation, transition Jacobians and process covariance.
- Observations own predictions, Jacobians, covariance and residual topology.
- Filters perform inference without owning model parameters.
- Random keys are explicit inputs to stochastic operations.

`blackwell._experiments` and names beginning with an underscore are private. The [API reference](/blackwell/api/) is the supported public surface.

## Documentation

The public documentation is published as part of the EICRL Hugo site for a consistent lab-wide reading experience. Exact code-level details also live in source docstrings and are checked by the Blackwell repository's local MkDocs build.

When public behaviour changes, update source docstrings, runnable examples and the corresponding guide or reference page in the lab site.

## Linear prediction benchmark

```console
uv run python benchmarks/linear_prediction.py --horizon 32 --batch-size 256
```

This compares prepared matrix products and a batched scan for identical affine
outputs and objective gradients, separating preparation/compilation from warmed
execution. See the [prediction guide](/blackwell/guides/linear-prediction/) for
storage, XLA buffer estimates and preparation break-even counts.

## Release state

The current public release is `0.0.4`, adding reusable linear predictions and
decision-relevant model evaluation to the existing estimation API. See the
[roadmap and release scope](/blackwell/help/roadmap/) for the wider v0.1 plan.
Releases use semantic versioning, with API stability expected only from version
1.0 onwards. Each release is built as a wheel and source distribution, verified
in clean Python 3.10–3.13 environments, published to PyPI through GitHub Actions
Trusted Publishing, and documented in the
[changelog](https://github.com/unswei/blackwell/blob/main/CHANGELOG.md).
