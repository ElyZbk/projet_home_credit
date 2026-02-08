"""Basic API tests to ensure the FastAPI app responds as expected."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api_app import app


def _load_fixture_payload() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    fixture_path = project_root / "tests" / "fixtures" / "sample_payload.json"
    assert fixture_path.exists(), (
        f"Missing fixture: {fixture_path}. Generate it from notebook 02."
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_health_endpoint():
    # Le "with" déclenche bien lifespan/startup/shutdown
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ("ok", "degraded")


def test_predict_endpoint_returns_expected_fields():
    payload = _load_fixture_payload()

    with TestClient(app) as client:
        # (optionnel mais utile) vérifier que le modèle est chargé
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json().get("model_loaded") is True, h.text

        r = client.post("/predict", json=payload)

    assert r.status_code == 200, r.text
    data = r.json()

    # Champs attendus
    assert data["status"] == "ok"
    assert "probability_default" in data
    assert "predicted_label" in data
    assert "decision" in data
    assert "threshold_used" in data

    # Vérifs simples
    p = data["probability_default"]
    y = data["predicted_label"]
    decision = data["decision"]
    thr = data["threshold_used"]

    assert isinstance(p, (float, int))
    assert 0.0 <= float(p) <= 1.0

    assert isinstance(y, int)
    assert y in (0, 1)

    assert decision in ("ACCEPTED", "REFUSED")

    assert isinstance(thr, (float, int))
    assert 0.0 < float(thr) < 1.0