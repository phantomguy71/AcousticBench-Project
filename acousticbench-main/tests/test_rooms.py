from echobench.rooms import alpha_for_t60, sabine_t60, synth_rir


def test_sabine_roundtrip():
    dims = (5.0, 4.0, 3.0)
    alpha = alpha_for_t60(dims, 0.6)
    assert abs(sabine_t60(dims, alpha) - 0.6) < 1e-9


def test_rir_shape_and_peak():
    rir = synth_rir(1.0, 16000, 0.5, 2.0, seed=3)
    assert rir.shape[0] == 16000
    assert float(abs(rir).max()) > 0.5
