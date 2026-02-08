"""Utilities to load the champion model and run inference with business thresholds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

# Support execution as module (`python -m src.inference`) and as script (`python src/inference.py`)
try:
    from . import config  # type: ignore
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import config  # type: ignore


class ChampionModel:
    """Wrapper around the trained pipeline plus metadata.

    Parameters
    ----------
    pipeline_path:
        Path to the persisted sklearn pipeline (joblib format).
    metadata_path:
        Path to the JSON file containing the decision threshold and any
        additional metadata (cost assumptions, version info, etc.).
    """

    def __init__(self, pipeline_path: Path, metadata_path: Path) -> None:
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.threshold = config.DEFAULT_THRESHOLD

    def load(self) -> "ChampionModel":
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {self.pipeline_path}")
        self.pipeline = joblib.load(self.pipeline_path)

        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.threshold = float(metadata.get("best_threshold", self.threshold))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_loaded()
        return self.pipeline.predict_proba(df)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        self._ensure_loaded()
        thresh = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(df)
        return (proba >= thresh).astype(int)

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Champion pipeline not loaded. Call load() first.")


def load_champion() -> ChampionModel:
    """Helper that loads the champion pipeline using default config paths."""
    model = ChampionModel(
        pipeline_path=config.CHAMPION_PIPELINE_PATH,
        metadata_path=config.THRESHOLD_METADATA_PATH,
    )
    return model.load()


def predict_single(payload: Dict[str, Any], *, model: ChampionModel | None = None) -> Dict[str, Any]:
    """Run inference on a single JSON payload.

    Returns the probability and business decision.
    """
    if model is None:
        model = load_champion()

    df = pd.DataFrame([payload])
    proba = float(model.predict_proba(df)[0])
    decision = int(proba >= model.threshold)
    return {
        "probability": proba,
        "decision": decision,
        "threshold": model.threshold,
    }
