"""
detection.py — WiFi traffic classification using trained model.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Detector:
    """
    Loads the ONNX model and classifies feature vectors extracted by features.py.

    Usage:
        detector = Detector("model/ids_xgb.onnx")
        label, confidence = detector.predict(feature_vector)
    """

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed: pip install onnxruntime")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}\n"
                "First run: python train.py --dataset <awid.csv>"
            )

        self._session = ort.InferenceSession(str(path))
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"Model incarcat: {path.name}")

    def predict(self, feature_vector: List[float]) -> Tuple[str, float]:
        """
        Classifies a single feature vector as "normal" or "abnormal" with confidence score.

        Args:
            feature_vector: list of 21 floats produced by features_to_vector()

        Returns:
            (label, confidence)
            label      — "normal" or "abnormal"
            confidence — probability 0.0–1.0 for the predicted label
        """
        x = np.array([feature_vector], dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: x})

        label_int  = int(outputs[0][0])
        confidence = float(outputs[1][0][label_int])

        return ("abnormal" if label_int == 1 else "normal"), confidence
