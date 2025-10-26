"""Benchmark harness utilities for the Persist evaluation suite."""

from __future__ import annotations

import copy
import itertools
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

import yaml

from main import run_experiment

MetricDict = Mapping[str, float]

__all__ = [
    "BaselineResult",
    "BenchmarkSweepReport",
    "SweepRunResult",
    "SweepSummary",
    "TMWRecoveryBenchmark",
]


@dataclass(frozen=True)
class BaselineResult:
    """Stores per-seed baseline metrics."""

    seed: int
    metrics: Dict[str, float]


@dataclass(frozen=True)
class SweepRunResult:
    """Captures metrics for a lineage-enabled sweep run."""

    seed: int
    metrics: Dict[str, float]
    delta_from_baseline: Dict[str, float]


@dataclass(frozen=True)
class SweepSummary:
    """Aggregated statistics for a specific (rank, ancestor) combination."""

    lora_rank: int
    max_ancestors: int
    mean_metrics: Dict[str, float]
    mean_delta_from_baseline: Dict[str, float]
    per_seed: List[SweepRunResult] = field(default_factory=list)
    improvement_confirmed: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSweepReport:
    """Full sweep report including baselines and aggregated summaries."""

    scenario: str
    metrics: Sequence[str]
    baselines: List[BaselineResult]
    sweeps: Mapping[str, List[SweepSummary]]


class TMWRecoveryBenchmark:
    """Orchestrates sweeps for the Transgenerational Memory Weave scenario."""

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        base_config_path: str | Path = "config.yaml",
    ) -> None:
        self.scenario_path = Path(scenario_path)
        if not self.scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario definition not found: {self.scenario_path}"
            )

        with self.scenario_path.open("r", encoding="utf-8") as handle:
            self.scenario = yaml.safe_load(handle)

        base_config_path = Path(base_config_path)
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")

        with base_config_path.open("r", encoding="utf-8") as handle:
            self.base_config = yaml.safe_load(handle)

        setup = self.scenario.get("setup") or {}
        metrics = setup.get("metrics") or []
        self.metric_names: List[str] = [
            entry["name"] for entry in metrics if "name" in entry
        ]

        runs = setup.get("runs") or {}
        if not runs:
            raise ValueError(
                "Scenario must define at least one run entry under setup.runs."
            )
        self.runs: Dict[str, Dict[str, object]] = runs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_sweep(
        self,
        *,
        lora_ranks: Sequence[int],
        ancestor_counts: Sequence[int],
        seeds: Sequence[int],
        run_experiment_fn: (
            Callable[[MutableMapping[str, object]], object] | None
        ) = None,
        metric_extractor: (
            Callable[[object, Mapping[str, object]], MetricDict] | None
        ) = None,
    ) -> BenchmarkSweepReport:
        """Execute the configured sweep across seeds and lineage parameters."""

        if not lora_ranks:
            raise ValueError("At least one LoRA rank must be provided.")
        if not ancestor_counts:
            raise ValueError("At least one ancestor count must be provided.")
        if not seeds:
            raise ValueError("At least one seed must be provided.")

        runner = run_experiment_fn or run_experiment
        extractor = metric_extractor or (lambda result, _: result)  # type: ignore[assignment]

        baseline_name = self._resolve_baseline_run_name()
        lineage_runs = self._resolve_lineage_run_names(excluding={baseline_name})
        if not lineage_runs:
            raise ValueError("Scenario must include at least one lineage-enabled run.")

        baseline_metrics = self._compute_baselines(
            baseline_name,
            seeds=seeds,
            runner=runner,
            extractor=extractor,
        )

        sweeps: Dict[str, List[SweepSummary]] = {}
        for run_name in lineage_runs:
            sweeps[run_name] = self._compute_lineage_sweeps(
                run_name,
                lora_ranks=lora_ranks,
                ancestor_counts=ancestor_counts,
                seeds=seeds,
                runner=runner,
                extractor=extractor,
                baseline_lookup={
                    result.seed: result.metrics for result in baseline_metrics
                },
            )

        return BenchmarkSweepReport(
            scenario=str(self.scenario_path),
            metrics=self.metric_names,
            baselines=baseline_metrics,
            sweeps=sweeps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_baselines(
        self,
        run_name: str,
        *,
        seeds: Sequence[int],
        runner: Callable[[MutableMapping[str, object]], object],
        extractor: Callable[[object, Mapping[str, object]], MetricDict],
    ) -> List[BaselineResult]:
        overrides = self.runs[run_name]
        results: List[BaselineResult] = []
        for seed in seeds:
            config = self._build_run_config(overrides, seed)
            outcome = runner(config)
            metrics = self._extract_metrics(extractor(outcome, config))
            results.append(BaselineResult(seed=seed, metrics=metrics))
        return results

    def _compute_lineage_sweeps(
        self,
        run_name: str,
        *,
        lora_ranks: Sequence[int],
        ancestor_counts: Sequence[int],
        seeds: Sequence[int],
        runner: Callable[[MutableMapping[str, object]], object],
        extractor: Callable[[object, Mapping[str, object]], MetricDict],
        baseline_lookup: Mapping[int, Dict[str, float]],
    ) -> List[SweepSummary]:
        overrides = self.runs[run_name]
        summaries: List[SweepSummary] = []

        for rank, ancestor_count in itertools.product(lora_ranks, ancestor_counts):
            per_seed_results: List[SweepRunResult] = []
            delta_accumulator: Dict[str, List[float]] = {
                name: [] for name in self.metric_names
            }
            metric_accumulator: Dict[str, List[float]] = {
                name: [] for name in self.metric_names
            }

            for seed in seeds:
                config = self._build_run_config(
                    overrides,
                    seed,
                    lineage_params={
                        "lora_rank": int(rank),
                        "max_ancestors": int(ancestor_count),
                    },
                )
                outcome = runner(config)
                metrics = self._extract_metrics(extractor(outcome, config))
                baseline_metrics = baseline_lookup[seed]
                deltas = {
                    metric: baseline_metrics.get(metric, 0.0) - metrics.get(metric, 0.0)
                    for metric in self.metric_names
                }

                for metric, value in metrics.items():
                    if metric in metric_accumulator:
                        metric_accumulator[metric].append(value)
                for metric, value in deltas.items():
                    if metric in delta_accumulator:
                        delta_accumulator[metric].append(value)

                per_seed_results.append(
                    SweepRunResult(
                        seed=seed,
                        metrics=metrics,
                        delta_from_baseline=deltas,
                    )
                )

            mean_metrics = {
                metric: statistics.fmean(values)
                for metric, values in metric_accumulator.items()
                if values
            }
            mean_deltas = {
                metric: statistics.fmean(values)
                for metric, values in delta_accumulator.items()
                if values
            }
            improvement_flags = {
                metric: all(value > 0 for value in values)
                for metric, values in delta_accumulator.items()
                if values
            }

            summaries.append(
                SweepSummary(
                    lora_rank=int(rank),
                    max_ancestors=int(ancestor_count),
                    mean_metrics=mean_metrics,
                    mean_delta_from_baseline=mean_deltas,
                    per_seed=per_seed_results,
                    improvement_confirmed=improvement_flags,
                )
            )

        return summaries

    def _build_run_config(
        self,
        overrides: Mapping[str, object],
        seed: int,
        lineage_params: Optional[Mapping[str, int]] = None,
    ) -> MutableMapping[str, object]:
        config = copy.deepcopy(self.base_config)
        config["seed"] = int(seed)
        self._deep_update(config, overrides)

        if lineage_params:
            lineage_section = config.setdefault("lineage", {})
            if not isinstance(lineage_section, dict):
                raise TypeError(
                    "Config lineage section must be a dictionary when overriding."
                )
            for key, value in lineage_params.items():
                lineage_section[key] = int(value)

        return config

    def _deep_update(
        self,
        target: MutableMapping[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        for key, value in overrides.items():
            if isinstance(value, Mapping):
                existing = target.get(key)
                if isinstance(existing, MutableMapping):
                    self._deep_update(existing, value)
                else:
                    target[key] = copy.deepcopy(value)
            else:
                target[key] = copy.deepcopy(value)

    def _extract_metrics(self, raw: Mapping[str, object]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name in self.metric_names:
            value = raw.get(name)
            if value is None:
                continue
            try:
                metrics[name] = float(value)
            except (TypeError, ValueError):
                continue
        return metrics

    def _resolve_baseline_run_name(self) -> str:
        for run_name, overrides in self.runs.items():
            lineage_cfg = overrides.get("lineage", {})
            enabled = False
            if isinstance(lineage_cfg, Mapping):
                enabled = bool(lineage_cfg.get("enabled", False))
            if not enabled:
                return run_name
        raise ValueError("No baseline run (with lineage disabled) found in scenario.")

    def _resolve_lineage_run_names(self, *, excluding: Iterable[str]) -> List[str]:
        excluded = set(excluding)
        lineage_runs: List[str] = []
        for run_name, overrides in self.runs.items():
            if run_name in excluded:
                continue
            lineage_cfg = overrides.get("lineage", {})
            if isinstance(lineage_cfg, Mapping) and lineage_cfg.get("enabled", False):
                lineage_runs.append(run_name)
        return lineage_runs
