import math

from benchmarks.persist_suite.harness import TMWRecoveryBenchmark


def test_tmw_recovery_benchmark_generates_sweep_results():
    harness = TMWRecoveryBenchmark("benchmarks/persist_suite/tmw_recovery.yaml")

    call_records = []

    def runner(config):
        call_records.append(config)
        lineage_cfg = config.get("lineage", {})
        seed = config["seed"]
        if not lineage_cfg.get("enabled", False):
            return {
                "reward_recovery_steps": 100.0 + seed,
                "unsafe_exploration_rate": 0.2,
            }
        lora_rank = lineage_cfg.get("lora_rank", 0)
        max_ancestors = lineage_cfg.get("max_ancestors", 0)
        return {
            "reward_recovery_steps": 90.0 - 2.0 * lora_rank + max_ancestors + seed,
            "unsafe_exploration_rate": 0.05 + 0.01 * max_ancestors,
        }

    report = harness.run_sweep(
        lora_ranks=[4],
        ancestor_counts=[2, 3],
        seeds=[1, 2],
        run_experiment_fn=runner,
        metric_extractor=lambda result, _: result,
    )

    assert report.metrics == [
        "reward_recovery_steps",
        "unsafe_exploration_rate",
    ]
    assert len(report.baselines) == 2
    assert len(report.sweeps) == 1

    sweep_entries = list(report.sweeps.values())[0]
    assert len(sweep_entries) == 2

    first_entry = sweep_entries[0]
    assert first_entry.lora_rank == 4
    assert first_entry.max_ancestors == 2
    assert math.isclose(
        first_entry.mean_delta_from_baseline["reward_recovery_steps"], 16.0
    )
    assert all(delta > 0 for delta in first_entry.improvement_confirmed.values())

    assert len(call_records) == 6  # 2 baseline runs + 4 sweep runs
