---
title: "API reference"
description: "Public Blackwell modules, containers, signatures, shapes and coordinate conventions."
weight: 50
lastmod: 2026-09-20T00:00:00+10:00
draft: false
---

Use this reference for exact public signatures and array shapes. Start with a [task-led guide](/blackwell/guides/) if you are learning the library.

| Area | Public modules |
| --- | --- |
| Beliefs | `blackwell.beliefs` |
| State spaces | `blackwell.spaces.euclidean`, `blackwell.spaces.se2`, [`blackwell.spaces.se3`](/blackwell/api/se3/) |
| Models | `blackwell.models.linear`, `blackwell.models.se2`, `blackwell.models.range_bearing` |
| Filters | `blackwell.filters.ekf`, `blackwell.filters.particle` |
| Experiment support | `blackwell.simulation`, `blackwell.metrics` |

## Import conventions

Principal belief containers are available at package level:

```python
from blackwell import GaussianBelief, ParticleBelief
```

Algorithms and operation families use explicit imports:

```python
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.spaces import se2
```

Only modules documented here are supported public API. Names under `blackwell._experiments` and `blackwell.filters._protocols` are private.

The reference includes the public API released in **0.0.3**. For implementation details and the most recent source docstrings, follow each page's source links.
