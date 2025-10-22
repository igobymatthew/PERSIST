# UI/UX Improvement Ideas for PERSIST

This document captures a living backlog of UI and UX enhancements that build on the existing Rich-powered CLI wizard, telemetry plumbing, and onboarding docs. The goal is to strengthen guidance, provide richer situational awareness during long-running training, and smooth the journey from first launch to advanced experimentation.

## CLI & TUI Experience
- **Adaptive step navigator**: Extend the `StepProgressTracker` so each step can expand with contextual help, example defaults, and quick doc links when users pause. Layer in lightweight tooltips and glossary pivots that appear when the user hesitates (e.g., empty submission) so learning never halts.
- **Rewindable navigation controls**: Introduce global commands like `/back`, `/summary`, or keyboard shortcuts that let operators revisit earlier decisions, audit their selections, and resume from any checkpoint without restarting the wizard.
- **Scenario search and presets**: Layer a searchable catalog of experiment templates into the opening wizard screen. Seed it with roadmap-inspired scenarios so newcomers can pick a recommended path while power users filter by goals like multi-agent resilience or biodiversity exploration.
- **Undo/branch workflows**: Offer a "checkpoint" command that snapshots current answers, letting users branch experiments or roll back a set of tweaks without repeating the full flow. This keeps the CLI forgiving for iterative tuning sessions.

## Run-Time Feedback & Visualization
- **Layered progress timelines**: Augment training logs with Rich-rendered timelines that annotate milestone badges, shield interventions, life-stage changes, and curriculum unlocks. Seeing events aligned with metric swings builds trust during long runs.
- **Celebratory milestone callouts**: Celebrate thresholds (first curriculum completion, shield saves) with animated banners or badge toasts so progress feels rewarding even in text-only sessions.
- **Live episode counters & gauges**: Wrap core training loops with progress bars or live counters (e.g., `rich.progress`) that broadcast episode counts, reward deltas, and estimated time remaining without drowning the log in noise.
- **Context-aware alerts**: When telemetry flags risk spikes, surface focused panels summarizing root metrics plus next-step suggestions (e.g., "tighten shield threshold" links) instead of generic warnings. Highlight contributing signals so triage feels actionable.
- **Config diff overlay at launch**: Before training starts, render a dual-pane diff comparing customized settings against the default template. Export or screenshot options make it easy to preserve the exact setup that produced a run for reproducibility.

## Telemetry & Dashboarding
- **Composable dashboards**: Assemble Prometheus-fed widgets into a modular TUI or lightweight web dashboard that tracks survival steps, life-stage indices, trophic stability, and shield activity. Include sparkline trends and goal lines sourced from `config.yaml` KPIs.
- **Goal-line configurators**: Mirror KPI targets stored in configuration files and let users raise or lower threshold lines from the dashboard itself, triggering CLI badges or alerts when goals are surpassed.
- **Session bookmarking & replay**: Allow operators to bookmark timestamps (best reward, first failure) and later replay telemetry slices alongside the CLI decisions that led to them, bridging operational insight with configuration context.
- **Alert routing profiles**: Provide toggleable notification profiles (quiet, balanced, verbose) so different operators can choose how aggressively telemetry events surface during training sessions.
- **Dashboard starter kit**: Ship a ready-to-run dashboard scaffold (TUI or minimal web app) that binds to the Prometheus endpoint so teams can stand up unified status boards without bespoke glue code.

## Onboarding & Knowledge Surfaces
- **Guided tour overlays**: Detect first-run usage and display micro-tooltips that connect each CLI step to walkthroughs and roadmap docs, with options to snooze or revisit later. Tie the tips to the persistence philosophy so choices feel grounded.
- **Interactive glossary drawer**: Embed a toggleable glossary that highlights jargon as it appears and pulls definitions from theory and training docs. Let users expand terms in-place so they never leave the terminal to decode vocabulary.
- **Visual journey maps**: Publish a flowchart-style journey map in the README and surface it from the CLI welcome panel so operators can see the full arc from setup to telemetry at a glance.
- **Annotated run transcripts**: Capture screenshots or TUI recordings of landmark walkthroughs and embed commentary that explains why each step matters, giving learners a visual reference alongside the textual guide.
- **Cross-medium progress tracker**: Mirror CLI achievements (e.g., "completed training setup," "launched telemetry") in documentation checklists so newcomers see concrete next steps and feel momentum across mediums.

## Accessibility & Personalization
- **Theme and layout presets**: Offer light, dark, and high-contrast palettes plus condensed or expanded layouts so users can tailor Rich output to their environment or assistive tools.
- **Input modality shortcuts**: Add keyboard shortcuts (`n`, `b`, `s`) and optional numbered menus so the CLI remains efficient on remote shells or for assistive-device workflows.
- **Animation controls**: Respect low-bandwidth and log aggregation contexts by letting users disable animated progress bars or switch to static glyph updates while still conveying state transitions clearly.

These enhancements aim to create a cohesive, confidence-inspiring experience that supports first-time explorers and veteran operators alike, ensuring every run remains observable, explainable, and reproducible.
