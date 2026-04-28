---
title: "research directions"
toc: true
date: 2025-12-14T18:27:00+11:00
---
Our long-term goal is to understand and engineer physical and collective intelligence in robots, so that robust autonomy emerges from the interaction of body, learning, and structure rather than from computational scale alone.

# Topics
<div class="research-topics-layout">
<div class="research-topics-list">
{{% toc %}}
</div>
<aside class="research-note-link" aria-label="Related note">
  <p class="research-note-kicker">Note</p>
  <a href="/notes/applying-for-a-phd-with-me/">Applying for a PhD with me</a>
  <p>Information for prospective doctoral applicants on supervision, admission, and scholarship funding.</p>
</aside>
</div>


## Data-efficient embodied learning for manipulation

Robots can manipulate objects in controlled settings, but learning robust skills that generalise is still slow and data-hungry. This research direction develops **data-efficient embodied learning for manipulation**, aiming to acquire new skills from limited real-world experience by combining learning with physically grounded models and optimisation.

A central theme is **uncertainty-aware learning and transfer**. The aim is to build pipelines that represent what the robot does not know, then use that uncertainty to decide what can be learned in simulation, what must be learned in the real world, and how to adapt safely during execution. This matters most in nonlinear, contact-rich manipulation, where simple approximations can fail.

Work in this direction includes:
- differentiable and model-based learning pipelines for manipulation, including gradient-based trajectory or policy optimisation
- sim-to-real transfer with explicit uncertainty modelling, including calibrated system identification and uncertainty-guided adaptation
- learning from demonstrations and teleoperation to reduce trial-and-error, followed by data-efficient policy refinement
- dexterous, contact-rich manipulation that tightly integrates perception, control, and learning, with skill representations that support transfer and generalisation

The direction typically combines modern simulation with targeted real-robot experimentation.

***

## Collective robotics under physical uncertainty

Teams of robots are increasingly expected to operate in places where sensing is noisy, dynamics are only partly known, and each robot has a narrow, local view. The core challenge is to achieve reliable coordination without assuming perfect models or continuous, high-bandwidth communication.

This research direction develops **decentralised methods** where each robot maintains an explicit notion of uncertainty and the group uses information intelligently. The emphasis is on treating **information as a limited resource**: deciding what to infer locally, what to share with others, and how to plan and learn so that behaviour remains dependable when real-world conditions differ from simulation.

Key themes include:
- decentralised belief representations and belief compression for scalable decision-making under partial observability
- communication-efficient coordination, including learned message policies and robustness to dropouts
- uncertainty-aware multi-agent learning with improved credit assignment and stability at scale
- information-based objectives for exploration and coordination, such as empowerment and value-of-information ideas

The main constraints in this area are well understood, but still not solved in a way that transfers cleanly to physical robot teams.

***

## Learning and uncertainty for physical systems

This research stream investigates how machine learning can support prediction, simulation, and decision-making in complex physical systems, where modelling is often challenged by uncertainty, partial observability, and limited data. We focus on hybrid approaches that combine physics-based models with learning-based components to improve reliability and generalisation.

Core themes include physics-informed neural networks, neural operators, graph-based surrogates, and differentiable programming. A particular emphasis is placed on uncertainty quantification and probabilistic methods that enable safe deployment in high-stakes applications.

Current work includes modelling and control of composite manufacturing processes (as part of a funded ARC Discovery Project), with broader relevance to robotics, scientific computing, and real-time monitoring. Typical challenges involve learning from sparse or noisy sensor data, reducing simulation costs, and integrating learned models into control or planning pipelines.

***
