Agent Development Guidelines

Required Checks
	•	✅ Run pytest from the project root before committing any changes to agent logic.
	•	✅ Verify type consistency with mypy agents/ and ruff check . before merges.
	•	✅ When editing shared models or signals, confirm downstream compatibility by running:

python -m agents.tests.integration_check


	•	✅ Run black . and ensure no docstring or comment drift (architecture diagrams stay synced).

⸻

Code Style Notes
	•	Keep each agent self-contained; avoid circular imports with shared utilities.
	•	Maintain stateless compute paths wherever possible—side effects only occur in controller or coordinator layers.
	•	When implementing new Agent.evaluate() methods:
	•	Always return typed dicts or dataclasses (no raw tuples).
	•	Include a "context" key for telemetry and post-hoc visualization.
	•	EEA agents: Clamp all affective values in [0.0, 1.0] before returning.
	•	MPC agents: Ensure cost matrices remain positive semi-definite; validate using np.all(np.linalg.eigvals(Q) >= 0).
	•	Persist agents: Keep reward shaping functions explicit (λ_H, λ_I, shield_α) — no hidden weighting logic.
	•	Centralize all randomness in a single seeded RNG (rng = np.random.default_rng(seed)).

⸻

Design Principles
	•	Keep temporal separation between sensing, modulation, and decision phases:
	1.	Sense → 2. Estimate → 3. Modulate → 4. Decide → 5. Learn
	•	Avoid passing raw state objects between layers. Always hand off normalized signals or structured telemetry.
	•	Each layer should have one clearly defined input–output contract (no shared global state).
	•	Never re-implement standard scientific routines — rely on NumPy, SciPy, or scikit-learn for numerical stability.
	•	Document every non-trivial formula with a short inline reference (e.g., # per Sutton & Barto (2018), Eq. 3.7).

⸻

Integration Expectations
	•	All agent outputs must be compatible with PersistAgent’s schema:

{
    "ratio": float,
    "behavior": str,
    "meaning": float,
    "equilibrium_prior": float,
    "context": Dict[str, float]
}


	•	Implement reset() for any agent maintaining buffers, momentum, or meta-learning priors.
	•	Ensure async-safe serialization (.state_dict(), .load_state_dict()) if models are torch-based.
	•	Provide lightweight .mock_eval() methods for fast offline validation without GPU.

⸻

Research & Reference Notes
	•	Document all adaptive constants (e.g., anticipation_gain, calibration_gain) in a local docstring with citations.
	•	Keep docs/EEA.md and docs/PERSIST.md synchronized when formulae evolve.
	•	For stochastic processes, maintain analytical expected values under zero-mean Gaussian noise.
	•	Where possible, model nonlinearities as smooth, differentiable functions (softsign, tanh, sigmoid) for future autodiff compatibility.

⸻

Ideas to Explore Later
	•	Implement inter-agent communication hooks for multi-policy negotiation.
	•	Add adaptive noise scaling based on Shannon entropy of behavior distributions.
	•	Prototype energy budget models for PERSIST agents (metabolic cost weighting).
	•	Explore meta-policy blending via temperature-controlled softmax between MPC and EEA outputs.
	•	Add telemetry dashboards to visualize valence, fear, and equilibrium trajectories in real time.