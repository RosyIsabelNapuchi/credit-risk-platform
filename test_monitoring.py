import numpy as np

from models.monitoring import calibration_status, psi, psi_status


def test_psi_same_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert psi(a, b) < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 20000), rng.normal(0.8, 1, 20000)) > 0.25


def test_psi_handles_binary_variables():
    a = np.array([0] * 800 + [1] * 200)
    b = np.array([0] * 500 + [1] * 500)
    assert psi(a, b) > 0.1


def test_status_thresholds():
    assert psi_status(0.05) == "estable"
    assert psi_status(0.15) == "vigilar"
    assert psi_status(0.30) == "alerta"


def test_calibration_status():
    assert calibration_status(0.08, 0.081, 1500) == "calibrado"
    assert calibration_status(0.08, 0.14, 1500) == "subestima riesgo"
    assert calibration_status(0.08, 0.03, 1500) == "sobreestima riesgo"
