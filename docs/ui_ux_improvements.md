# UI/UX Improvement Ideas for PERSIST

This document captures potential enhancements to improve the framework's user experience, with an emphasis on providing clearer visual cues for progression throughout configuration and training workflows.

## 1. Guided CLI Enhancements
- **Step Progress Indicator**: Add a persistent header or sidebar-style printout in `main.py` that shows the user's current step within the experiment setup wizard (e.g., `1/5: Select Experiment Type`). This gives immediate context about progress and remaining tasks.
- **Color-Coded Prompts**: Leverage libraries such as `rich` or `blessed` to apply consistent coloring (green for completed steps, yellow for current inputs, grey for pending items) so users can visually parse state at a glance.
- **Inline Validation Feedback**: When users input hyperparameters (e.g., epochs, learning rates), show real-time validation with icons like `✔`/`✖` and short tooltips to confirm acceptance or highlight issues.
- **Configuration Summary Cards**: After each major section, render a bordered summary block that reiterates the selections made so far, reinforcing progress before moving forward.

## 2. Training Progress Visualization
- **ASCII Progress Bars**: Integrate progress bars for episode loops and update steps (e.g., via `tqdm` or `rich.progress`) so training status is always visible, especially for long-running runs.
- **Milestone Badges**: Emit badge-like messages (e.g., `🏁 Episode 100`, `🛡️ Shield Accuracy > 90%`) when key metrics cross thresholds. This creates celebratory checkpoints that keep operators engaged.
- **Timeline of Events**: Extend the logging output to include a condensed timeline that captures notable events (shield activations, maintenance visits, adversarial encounters) in chronological order.
- **Alert Highlighting**: When the `TelemetryManager` raises alerts, render them with bold headers and color accents to distinguish urgent issues from routine logs.

## 3. Dashboard and Telemetry Upgrades
- **Unified Status Board**: Build a lightweight dashboard (web or TUI) that aggregates metrics from `ops/telemetry.py`, showing gauges for survival time, constraint violations, and shield trigger rates alongside sparkline trends.
- **Scenario Progress Maps**: For grid-based environments, add an optional mini-map that marks the agent's trajectory and critical events, helping users visualize spatial progress.
- **Configurable KPI Targets**: Allow users to define success thresholds in `config.yaml` and visualize them as goal lines across charts, making it obvious how close the agent is to desired performance levels.
- **Session Bookmarks**: Provide the ability to bookmark significant training states (e.g., best validation score) and annotate them for later review within the dashboard timeline.

## 4. Documentation and Onboarding Touchpoints
- **Quickstart Flowchart**: Include a flowchart or infographic in the README illustrating the end-to-end workflow from environment setup to telemetry review, reinforcing the progression narrative.
- **Annotated Sample Run**: Publish a walkthrough transcript that highlights each CLI step with accompanying screenshots or TUI captures, showing how visual cues evolve as the user advances.
- **UX Checklist**: Maintain a checklist in the docs for contributors describing required progression cues (progress bars, milestone markers, summaries) so UI consistency persists as features evolve.

## 4.5 Beginner-Centric Onboarding Enhancements
- **First-Run Guided Tour**: Detect when a user launches `main.py` for the first time and overlay contextual hints (e.g., "This wizard configures your first persistence experiment") that reference the core workflow described in `docs/roadmap.md`. Provide shortcuts to skip or replay the tour so newcomers can revisit explanations later.
- **Scenario Templates Library**: Bundle a curated "Getting Started" library of experiment presets that mirror the staged implementation path from `docs/theory.md` (Phase 1 → Phase 3). Each template should include inline callouts explaining why certain hyperparameters matter for persistence so beginners can learn by tweaking safe defaults.
- **Interactive Glossary Drawer**: Embed a toggleable glossary in the CLI or dashboard that surfaces definitions for persistence-specific terms (e.g., viability set, shield, near-boundary buffer) pulled from `docs/theory.md` and `docs/training_resilience.md`. Highlight glossary terms when they appear in prompts so new users can expand them in-place.
- **Practice Mode Sandbox**: Offer a low-stakes simulation mode that replays recorded demonstrations with narration explaining key decisions (aligning with the roadmap's suggestion to highlight single-agent foundations before multi-agent complexity). Pair the narration with visual markers for homeostatic rewards, shield interventions, and fire recovery tactics so learners can map theory to behavior.
- **Onboarding Progress Tracker**: Add a persistent checklist that spans documentation and CLI milestones (e.g., "Read Quickstart Flowchart", "Run practice sandbox", "Launch first training run"). Synchronize completion states with telemetry logs so beginners receive gentle reminders about recommended next steps without feeling overwhelmed.

## 5. Accessibility and Customization
- **Configurable Themes**: Offer light/dark and high-contrast themes for the CLI/dashboard to accommodate different lighting conditions and accessibility needs.
- **Toggleable Animations**: Allow users to enable or disable animated progress bars or transitions to support environments where static output is preferred (e.g., log aggregation systems).
- **Keyboard Shortcuts**: Introduce shortcuts (e.g., `n` for next, `b` for back, `s` for summary) within the CLI to accelerate navigation while keeping progression cues synchronized with user actions.

These enhancements aim to make experimentation with PERSIST more intuitive, motivational, and transparent, ensuring users always understand where they are in a workflow and what comes next.
