---
title: "projects"
toc: true
date: 2026-02-08T17:24:00+11:00


---
## muesli-bt

muesli-bt is a deliberately anachronistic-looking robotics behaviour system: a small Lisp at the centre, used to author behaviour trees that run inside a host robot application.

It is old-school in how you write behaviours (simple, explicit, hackable), but paired with more contemporary runtime expectations like inspection-ready logs, deterministic replay, and clear tick and deadline semantics. The process has been a bit anachronistic too: started manually, then finished off with some LLM-assisted vibe coding.

Technically, muesli-bt is an embeddable runtime for Lisp-authored behaviour trees, with bounded-time planning during ticks and explicit async calls (including cancellation and safe fallback patterns). It is designed to integrate cleanly into C++ hosts and to support tooling that can inspect, replay, and debug real robot runs.

- docs: https://unswei.github.io/muesli-bt/
- code: https://github.com/unswei/muesli-bt


