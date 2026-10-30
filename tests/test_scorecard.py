import pytest

from utils.scorecard import band_for, pd_to_score, recommended_limit


def test_base_point():
    # odds 30:1  ->  PD = 1/31  ->  600 puntos
    assert pd_to_score(1 / 31) == 600


def test_pdo_doubles_odds():
    # odds 60:1 -> +20 puntos
    assert pd_to_score(1 / 61) == 620


def test_score_decreases_with_pd():
    assert pd_to_score(0.01) > pd_to_score(0.05) > pd_to_score(0.2)


@pytest.mark.parametrize("score,band", [(700, "A"), (625, "A"), (624, "B"), (600, "B"), (590, "C"), (560, "D"), (400, "E")])
def test_bands(score, band):
    assert band_for(score)[0] == band


def test_limit_policy():
    assert recommended_limit(100_000, "A") == 100_000
    assert recommended_limit(100_000, "E") == 0
