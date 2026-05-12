"""
detection.py — Clasificare trafic WiFi folosind modelul ONNX antrenat.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Detector:
    """
    Incarca modelul ONNX si clasifica vectori de features extrasi de features.py.

    Usage:
        detector = Detector("model/ids_xgb.onnx")
        label, confidence = detector.predict(feature_vector)
    """

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime nu e instalat: pip install onnxruntime")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model ONNX negasit: {path}\n"
                "Ruleaza mai intai: python train.py --dataset <awid.csv>"
            )

        self._session = ort.InferenceSession(str(path))
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"Model incarcat: {path.name}")

    def predict(self, feature_vector: List[float]) -> Tuple[str, float]:
        """
        Clasifica o fereastra de trafic.

        Args:
            feature_vector: lista de 21 float-uri produsa de features_to_vector()

        Returns:
            (label, confidence)
            label      — "normal" sau "abnormal"
            confidence — probabilitate 0.0–1.0 pentru label-ul prezis
        """
        x = np.array([feature_vector], dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: x})

        label_int  = int(outputs[0][0])
        confidence = float(outputs[1][0][label_int])

        return ("abnormal" if label_int == 1 else "normal"), confidence
