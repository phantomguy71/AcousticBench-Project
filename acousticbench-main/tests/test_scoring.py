from echobench.scoring import classify_metrics, detect_metrics, regress_metrics
from echobench.util import paired_bootstrap_gap


def test_classification_basic():
    m = classify_metrics(["a", "a", "b"], ["a", "b", "b"], [0.9, 0.6, 0.8], [5.0, 6.0, 7.0])
    assert m["n_evaluated"] == 3
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
    assert m["confusion_matrix"]["counts"][0][0] == 1


def test_regression_basic():
    m = regress_metrics([1.0, 2.0], [1.1, 1.9], [1.0, 2.0], ["x", "x"], task="t60", unit="seconds")
    assert abs(m["mae"] - 0.1) < 1e-9
    assert m["by_condition"]["x"]["n"] == 2


def test_events_perfect():
    truth = [[(0.5, 0.7)]]
    m = detect_metrics(truth, [[(0.55, 0.75)]], tol=0.25, lat=[3.0])
    assert m["event_f1"] == 1.0


def test_bootstrap_gap():
    b = paired_bootstrap_gap([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], rounds=50, seed=0)
    assert b["gap"] == 1.0
    assert b["ci_low"] <= 1.0 <= b["ci_high"]
