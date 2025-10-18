# Issue: Expand Persistence Test Coverage and Wire Up CI

## Summary
Sections 4 and 5 of `docs/roadmap.md` emphasize the need for broader automated testing (multi-agent stepping, shield invariants, curriculum schedules) and continuous integration so regressions are caught early. The repository currently lacks GitHub Actions workflows and only has limited tests (`tests/test_sac_isolated.py`, smoke tests). This issue establishes the required coverage and automation.

## Implementation Plan
1. **Test suite expansion**
   - Add multi-agent environment tests exercising `environments/multi_agent_gridlife.py` (spawn agents, simulate collisions, verify termination semantics).
   - Create shield invariants tests ensuring `components/shield.py` never returns unsafe actions given mocked models.
   - Add curriculum scheduler tests verifying constraint tightening over time.
   - Cover schema validation failure cases in `tests/test_config_validation.py`.
2. **Benchmark markers**
   - Introduce pytest markers (e.g., `@pytest.mark.benchmark`) that run longer benchmarks from `benchmarks/` when explicitly requested.
3. **Continuous integration workflow**
   - Add `.github/workflows/ci.yml` that sets up Python, installs dependencies (via `requirements.txt` or `environment.yml`), runs `ruff check .`, `black --check`, `pytest`, and mypy for agents.
   - Configure caching for pip or conda to keep builds fast.
4. **Developer tooling**
   - Document the new workflow in `README.md` and add badges reflecting CI status.
   - Provide `make` targets or scripts (`scripts/run_ci_checks.sh`) to mirror the CI steps locally.

## Acceptance Criteria
- New pytest modules cover multi-agent stepping, shield invariants, curriculum schedules, and schema validation failures.
- Benchmark markers allow optional heavy runs without slowing default CI.
- GitHub Actions workflow runs linting, type checks, and pytest on every push/PR.
- README documents how to run the checks locally and links to the CI badge.
- All checks pass on the main branch once merged.
