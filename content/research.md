---
title: "Research directions"
description: "Research directions in embodied learning, collective robotics and intelligent physical systems at EICRL."
toc: false
date: 2025-12-14T18:27:00+11:00
---
## Research vision

### Persistent and cumulative embodied intelligence

Our long-term goal is to understand how physical agents accumulate, revise and transfer knowledge across robots, tasks, lifetimes and embodiments. They must be able to keep acting and recover as their bodies, sensors, capabilities and environments change.

We ask when experience remains valid after physical change, when new capabilities make existing representations inadequate, and when an agent must acquire missing knowledge before safe recovery becomes impossible. We study these questions through learning, reasoning, control and real-world experiments.

## Three connected research themes

<div class="research-theme-grid">
  <article class="research-theme-card">
    <p class="research-theme-number">01</p>
    <h3><a href="#data-efficient-embodied-learning-for-manipulation">Embodied learning</a></h3>
    <p>Data-efficient learning and manipulation grounded in real robot experience.</p>
  </article>
  <article class="research-theme-card">
    <p class="research-theme-number">02</p>
    <h3><a href="#collective-robotics-under-physical-uncertainty">Collective robotics</a></h3>
    <p>Reliable coordination for robot teams operating with uncertainty and limited communication.</p>
  </article>
  <article class="research-theme-card">
    <p class="research-theme-number">03</p>
    <h3><a href="#learning-and-uncertainty-for-physical-systems">Physical intelligence</a></h3>
    <p>Learning, inference and control for complex physical systems with limited data.</p>
  </article>
</div>

<aside class="research-join-callout" aria-labelledby="research-join-title">
  <div>
    <p class="research-join-kicker">Work with us</p>
    <h2 id="research-join-title">Interested in research with EICRL?</h2>
    <p>See current supervision availability, funding expectations and what to include in an enquiry.</p>
  </div>
  <a class="content-action" href="/join/">Join the lab</a>
</aside>


## Data-efficient embodied learning for manipulation

Robots can manipulate objects in controlled settings, but learning skills that hold up across different objects and conditions is still slow and data-hungry. We combine learning with physically grounded models and optimisation to help robots acquire manipulation skills from limited real-world experience.

We study **uncertainty-aware learning and transfer**: representing what a robot does not know and using that uncertainty to decide what it can learn in simulation, what requires real-world experience, and how to adapt safely during execution. This matters especially in nonlinear, contact-rich manipulation, where simple approximations can fail.

Work in this direction includes:
- differentiable and model-based learning pipelines for manipulation, including gradient-based trajectory or policy optimisation
- sim-to-real transfer with explicit uncertainty modelling, including calibrated system identification and uncertainty-guided adaptation
- learning from demonstrations and teleoperation to reduce trial-and-error, followed by data-efficient policy refinement
- dexterous, contact-rich manipulation that tightly integrates perception, control, and learning, with skill representations that support transfer and generalisation

We combine simulation with targeted experiments on real robots.

Related projects: [L1-GT](/projects/l1-gt/) · [Robotic assistance in physical care](/projects/assistive-care/)

***

## Collective robotics under physical uncertainty

Robot teams must coordinate when sensing is noisy, dynamics are only partly known, and each robot has a limited, local view. We study how to make that coordination reliable without assuming perfect models or continuous, high-bandwidth communication.

We develop **decentralised methods** in which each robot represents its uncertainty and treats **information as a limited resource**. Each robot must decide what to infer locally, what to share with others, and how to plan and learn when real-world conditions differ from simulation.

Key themes include:
- decentralised belief representations and belief compression for scalable decision-making under partial observability
- communication-efficient coordination, including learned message policies and robustness to dropouts
- uncertainty-aware multi-agent learning with improved credit assignment and stability at scale
- information-based objectives for exploration and coordination, such as empowerment and value-of-information ideas

Making these methods work reliably on physical robot teams remains an open problem.

Related projects: [muesli-bt](/projects/muesli-bt/) · [L1-GT](/projects/l1-gt/)

***

## Learning and uncertainty for physical systems

We study how machine learning can support prediction, simulation and decision-making in physical systems where data are limited, measurements are uncertain, and parts of the system cannot be observed directly. We combine physics-based models with learned components to improve reliability and generalisation.

Methods include physics-informed neural networks, neural operators, graph-based surrogates and differentiable programming. We use uncertainty quantification and probabilistic methods to assess when predictions can be trusted and support safe deployment.

Current work includes modelling and control of composite manufacturing processes as part of a funded ARC Discovery Project. This involves learning from sparse or noisy sensor data, reducing simulation costs, and integrating learned models into control or planning. These methods also have applications in robotics, scientific computing and real-time monitoring.

Related projects: [Smart manufacturing for composite structures](/projects/smart-manufacturing/) · [Five-bar linkage design tools](/projects/fivebar/)

***
