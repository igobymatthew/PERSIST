# 🧬 PERSIST vXR — Adaptive Ecosystem Engine

> 

PERSIST vXR is a research framework for **self-regulating, persistence-driven AI systems**.  
It treats learning, memory, and adaptation as *living ecological processes* rather than linear optimizations.

---

## 🌍 Overview

Instead of maximizing reward alone, PERSIST agents strive to **remain viable** within dynamic environments.  
They balance **empowerment** (control over future states) with **homeostasis** (stability within limits) and **evolutionary regeneration** (Fire Cycle events).  
The result is an ecosystem where agents, populations, and even the training engine itself can **self-adjust, self-repair, and evolve**.

---

## 🧠 Core Principles

| Principle | Description |
|------------|-------------|
| **Viability** | Agents maintain internal variables within safe bounds — the software analog of physiological integrity. |
| **Empowerment** | Intrinsic motivation to sustain control and reduce uncertainty. |
| **Homeostasis** | Continuous balancing of reward, energy, and constraint costs. |
| **Fire Cycles** | Periodic regeneration that prunes, compresses, and retrains for long-term stability. |
| **Evolutionary Adaptation** | Genetic algorithms and LoRA fine-tuning evolve architectures and behaviors. |
| **Ecosystem Simulation** | Multi-agent, multi-species environments model cooperation, competition, and ecological balance. |

---

## ⚙️ System Architecture

**Core Layers**

1. **Viability Kernel** — Predicts safety margins and filters unsafe actions.  
2. **Empowerment Engine** — Calculates intrinsic control and novelty rewards.  
3. **Homeostat / Constraint Manager** — Regulates internal energy and penalty multipliers.  
4. **Dreamer-style World Model** — Generates imagination rollouts for planning.  
5. **Meta-Learner & Fire Cycle** — Adjusts setpoints, triggers regeneration.  
6. **Overseer-LoRA** — Fine-tuned control model that manages other models.  
7. **Biodiversity Fabric Simulator (BFS)** — Full ecosystem with trophic stability and species richness.  

**Telemetry Stack:** Prometheus metrics, Grafana dashboards, and resilience analytics.

---

## 🔄 Current Capabilities

- Continual learning with EWC and rehearsal buffers  
- Fire Cycle regeneration with controlled entropy reduction  
- Multi-agent and population-level simulations (CTDE)  
- Dreamer-RSSM imagination rollouts  
- Emotional Equilibrium Agents (EEA++) with life-stage awareness  
- Evolutionary search via GA/NSGA-II engine  
- Prometheus telemetry and alerting for viability metrics  
- Biodiversity Fabric Simulator (BFS) for multi-species ecosystems  

---

## 📈 Roadmap Highlights

| Phase | Focus | Status |
|-------|--------|--------|
| **Core Stability** | Modularization, Viability Kernel, Empowerment loop | ✅ Complete |
| **Adaptive Reflection Layer** | Overseer-LoRA, telemetry integration | 🧩 In progress |
| **Fire Ecology** | Fire Cycle protocols, resilience metrics | 🔬 Testing |
| **Ecosystem Integration** | BFS mutualism & trophic balance | 🧪 Active |
| **Autopoietic Governance (vXR)** | Self-governing engine with bounded config rewrites | 🚧 Upcoming |

---

## 📚 Key References

- **Hafner et al., 2024 — DreamerV4:** latent imagination for continual adaptation.  
- **Friston & Parr, 2025 — Free Energy and Viability:** self-evidencing agents.  
- **Schmidhuber, 2026 — Intrinsic Curiosity Revisited:** Powerplay & empowerment.  
- **Maturana & Varela — Autopoiesis:** self-maintenance as life’s essence.  

---

## 🧩 Repository Layout

Core Directory Layout

persist/engine/ — Core logic for the adaptive persistence framework
	•	core/ — Base runtime, state/action mechanics, viability ops
	•	empowerment/ — Intrinsic motivation & control estimation modules
	•	viability/ — ViabilityApproximator, EnsembleShield, constraint functions
	•	memory/ — Rehearsal buffers, Elastic Weight Consolidation, persistence utilities

persist/agents/ — Implementations of behavioral and reflective agents
	•	mpc_agent.py — Reach-Avoid MPC controller
	•	cvar_sac.py — Risk-sensitive SAC variant (CVaR objective)
	•	emotional_eq.py — Emotional Equilibrium Agent++ (affect-based regulation)
	•	meta_learner.py — Meta-adaptive setpoint tuner

persist/population/ — Multi-agent and ecosystemic coordination layer
	•	multi_agent_env.py — Collective training environment
	•	ensemble_shield.py — Multi-model safety consensus layer
	•	biodiversity_fabric.py — Biodiversity Fabric Simulator (BFS) for ecosystem simulation

persist/components/ — Shared building blocks and computational primitives
	•	empowerment.py — Mutual information estimators, curiosity drivers
	•	constraint_manager.py — Penalty scaling and viability management
	•	safety_network.py — Viability and reachability classifiers
	•	latent_world_model.py — Dreamer-style RSSM imagination system
	•	cbf_layer.py — Control Barrier Function layer for constrained safety

persist/ops/ — Operational systems, telemetry, and observability
	•	telemetry.py — Prometheus hooks, metrics pipelines
	•	alerts.yml — Observability and failure alert configuration
	•	maintenance_tasks.py — Health checks and automated FireCycle triggers

persist/tools/ — Utilities for optimization, reachability, and experimentation
	•	ga_engine.py — Genetic Algorithm / NSGA-II evolutionary engine
	•	hj_reachability/ — Hamilton–Jacobi reachability utilities
	•	fuzz_scenarios.py — Stress-testing and perturbation scenario generator

persist/docs/ — Research documentation and conceptual references
	•	roadmap_vXR.md — Full academic + philosophical roadmap
	•	summary.md — GitHub front-page overview
	•	theory.md — Formal theoretical and mathematical background

persist/tests/ — Verification, regression, and safety tests
	•	test_components.py — Unit tests for low-level components
	•	test_viability.py — Viability kernel validation
	•	test_multiagent_env.py — BFS and population behavior tests

main.py — Interactive CLI entrypoint and experimental launcher

⸻

🧠 Contributor Notes

All modules follow a viability-first design philosophy:
	•	Each subsystem must expose Prometheus telemetry for self-monitoring.
	•	New features require rollback or FireCycle-safe reset logic.
	•	Configurations must remain reproducible through config.yaml manifests.
	•	All agents, even experimental ones, must integrate the Viability Kernel.

⸻
