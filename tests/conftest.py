import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Entrena un modelo rápido si no existe el artefacto, para que los tests sean autocontenidos."""
    from models.scoring import ARTIFACT, save, train
    from data.generate import make_applications

    if not ARTIFACT.exists():
        save(train(make_applications(n=3000))["xgb"])
