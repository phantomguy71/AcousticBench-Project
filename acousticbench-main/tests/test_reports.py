from echobench.comparison import compare_pair
from echobench.reporting import compare_markdown, leaderboard_rows, regression_alerts, run_markdown


def _run(rid, task, domain, metrics, model="m"):
    return {"id": rid, "status": "completed", "model_id": model, "model_name": model,
            "dataset_id": "d-" + domain, "dataset_name": "D", "dataset_domain": domain,
            "notes": "", "created_at": rid, "config": {"sample_rate": 16000, "dataset_flags": {}},
            "summary": {"task": task, **metrics}, "provenance": {}}


def test_compare_and_golden_text():
    sim = _run("a", "room", "simulated", {"accuracy": 0.8, "macro_f1": 0.75, "macro_precision": 0.8,
                                          "macro_recall": 0.7, "weighted_f1": 0.76, "ece": 0.05,
                                          "mean_latency_ms": 4.0, "n_evaluated": 20,
                                          "per_class": {"office": {"f1": 0.8, "support": 10}},
                                          "item_scores": [1, 1, 0]})
    real = _run("b", "room", "real", {"accuracy": 0.6, "macro_f1": 0.55, "macro_precision": 0.6,
                                      "macro_recall": 0.5, "weighted_f1": 0.56, "ece": 0.09,
                                      "mean_latency_ms": 5.0, "n_evaluated": 20,
                                      "per_class": {"office": {"f1": 0.6, "support": 10}},
                                      "item_scores": [1, 0, 0]})
    comp = compare_pair(sim, real)
    assert comp["gap_definition"].startswith("gap")
    text = compare_markdown(comp, sim, real)
    assert "# AcousticBench sim-to-real comparison" in text
    assert "Bootstrap" in text
    assert run_markdown(sim).startswith("# AcousticBench run report")


def test_leaderboard_and_regressions():
    r1 = _run("2026-01-01T00:00:00", "t60", "simulated", {"mae": 0.2, "rmse": 0.25, "n_evaluated": 12})
    r2 = _run("2026-01-02T00:00:00", "t60", "simulated", {"mae": 0.1, "rmse": 0.12, "n_evaluated": 12})
    board = leaderboard_rows([r1, r2])
    assert board[0]["run_id"] == r2["id"]
    assert regression_alerts([r2, r1]) == []  # newest is better -> no alert


def test_significance_block_present():
    sim = _run("a", "t60", "simulated", {"mae": 0.2, "rmse": 0.25, "median_ae": 0.2,
                                         "mean_latency_ms": 3.0, "n_evaluated": 12, "item_scores": [-0.2] * 12})
    real = _run("b", "t60", "real", {"mae": 0.3, "rmse": 0.35, "median_ae": 0.3,
                                     "mean_latency_ms": 3.5, "n_evaluated": 12, "item_scores": [-0.3] * 12})
    comp = compare_pair(sim, real)
    assert comp["significance"]["bootstrap"]["rounds"] == 2000
