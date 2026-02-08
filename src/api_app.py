"""FastAPI entrypoint exposing a /predict endpoint."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict

# Support running as module (`uvicorn src.api_app:app`) and as script (`python src/api_app.py`)
try:
    from .inference import load_champion, predict_single  # type: ignore
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.inference import load_champion, predict_single  # type: ignore

class ScoringPayload(BaseModel):
    """
    Schema pour requêtes /predict.

    - Champs supplémentaires acceptés (features du modèle).
    - SK_ID_CURR optionnel (traçabilité).
    """
    model_config = ConfigDict(extra="allow")  # ✅ Pydantic v2

    SK_ID_CURR: Optional[int] = Field(None, description="Unique customer identifier (optional)")


def _payload_to_dict(payload: ScoringPayload) -> Dict[str, Any]:
    return payload.model_dump()  # ✅ Pydantic v2


def _build_shap_explainer(model):
    """Build a SHAP TreeExplainer from the pipeline's XGBoost classifier."""
    classifier = model.pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


def _get_transformed_feature_names(model) -> List[str]:
    """Retrieve feature names after the ColumnTransformer preprocessing step."""
    preprocessor = model.pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le modèle au démarrage. Si échec, on démarre quand même et /health sera degraded.
    """
    try:
        app.state.champion_model = load_champion()
        app.state.model_load_error = None
        app.state.shap_explainer = _build_shap_explainer(app.state.champion_model)
    except FileNotFoundError as exc:
        app.state.champion_model = None
        app.state.model_load_error = str(exc)
        app.state.shap_explainer = None
    yield
    # shutdown: rien


app = FastAPI(
    title="Home Credit Scoring API",
    version="1.0.0",
    description="Predict probability of default and business decision (threshold-based).",
    lifespan=lifespan,
)


def get_model():
    model = getattr(app.state, "champion_model", None)
    if model is None:
        msg = getattr(app.state, "model_load_error", "Champion model not loaded")
        raise FileNotFoundError(msg)
    return model


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Home Credit Scoring API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        _ = get_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.post("/predict")
def predict(payload: ScoringPayload) -> Dict[str, Any]:
    try:
        model = get_model()
        data = _payload_to_dict(payload)

        # Traçabilité
        sk_id = data.get("SK_ID_CURR")
        data.pop("SK_ID_CURR", None)

        result = predict_single(data, model=model)
        decision_label = "REFUSED" if int(result["decision"]) == 1 else "ACCEPTED"

        return {
            "status": "ok",
            "sk_id_curr": sk_id,
            "probability_default": float(result["probability"]),
            "predicted_label": int(result["decision"]),
            "decision": decision_label,
            "threshold_used": float(result["threshold"]),
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input payload: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(exc).__name__}: {exc}") from exc


@app.post("/shap")
def shap_local(payload: ScoringPayload, top_n: int = 10) -> Dict[str, Any]:
    """Return local SHAP values for a single client (top N features)."""
    try:
        model = get_model()
        explainer = getattr(app.state, "shap_explainer", None)
        if explainer is None:
            raise HTTPException(status_code=503, detail="SHAP explainer not available")

        data = _payload_to_dict(payload)
        sk_id = data.pop("SK_ID_CURR", None)

        df = pd.DataFrame([data])
        preprocessor = model.pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(df)

        shap_values = explainer.shap_values(X_transformed)
        # shap_values shape: (1, n_features) for binary classification
        if isinstance(shap_values, list):
            # Some explainers return [class_0, class_1]; take class 1
            sv = np.array(shap_values[1]).flatten()
        else:
            sv = np.array(shap_values).flatten()

        feature_names = _get_transformed_feature_names(model)
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            base_value = float(expected[1])
        else:
            base_value = float(expected)

        # Sort by absolute SHAP value, descending
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        top_features = [
            {"feature": feature_names[i], "shap_value": float(sv[i])}
            for i in indices
        ]

        return {
            "status": "ok",
            "sk_id_curr": sk_id,
            "base_value": base_value,
            "top_shap": top_features,
        }

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP error: {type(exc).__name__}: {exc}") from exc