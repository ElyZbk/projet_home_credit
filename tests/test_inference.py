"""Unit tests for inference helper functions."""
import json
import sys
from pathlib import Path
from src.inference import  load_champion, predict_single

# Allow running this test file directly without pytest picking up conftest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



def test_predict_single_returns_valid_outputs():
    # Charger fixture JSON
    project_root = Path(__file__).resolve().parents[1]
    fixture_path = project_root / "tests" / "fixtures" / "sample_payload.json"

    assert fixture_path.exists(), f"Missing fixture: {fixture_path}. Generate it from notebook 02."

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Charger modèle champion
    model = load_champion()

    # Prédire
    result = predict_single(payload, model=model)

    # Vérifs structure
    assert "probability" in result
    assert "decision" in result
    assert "threshold" in result

    # Vérifs types / bornes
    p = result["probability"]
    d = result["decision"]
    t = result["threshold"]

    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0

    assert isinstance(d, int)
    assert d in (0, 1)

    assert isinstance(t, float)
    assert 0.0 < t < 1.0
