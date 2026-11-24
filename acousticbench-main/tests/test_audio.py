import numpy as np

from echobench.audio_io import read_mono_wav, store_wav, to_rate


def test_roundtrip(tmp_path):
    wav = tmp_path / "a.wav"
    sig = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.5
    store_wav(wav, sig, 16000)
    back, rate = read_mono_wav(wav)
    assert rate == 16000
    assert back.shape[0] == 16000
    assert float(np.max(np.abs(back - sig))) < 0.01


def test_resample_changes_length():
    sig = np.ones(16000)
    out = to_rate(sig, 16000, 8000)
    assert out.shape[0] == 8000
    assert to_rate(sig, 16000, 16000).shape[0] == 16000
