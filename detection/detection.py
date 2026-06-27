"""
detection.py - Clasificare trafic WiFi folosind modelul ONNX antrenat.
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

    def __init__(self, model_path: str, threshold: float = 0.5):
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

        self.threshold = threshold
        self._session = ort.InferenceSession(str(path))
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"Model incarcat: {path.name} (prag decizie={threshold:.2f})")

    def predict(self, feature_vector: List[float]) -> Tuple[str, float]:
        """
        Clasifica o fereastra de trafic.

        Decizia se ia pe P(atac) >= prag, NU pe argmax. Asta permite coborarea
        pragului pentru a prinde mai multe atacuri (recall mare / false negatives
        mici) - un argmax fix la 0.5 ar bloca acest reglaj. Pragul optim se afla
        cu sweep-ul din evaluate.py (--target-recall).

        Args:
            feature_vector: lista de 12 float-uri produsa de features_to_vector()

        Returns:
            (label, prob_atac)
            label    - "abnormal" daca P(atac) >= prag, altfel "normal"
            prob_atac - probabilitatea de atac 0.0-1.0 (scorul pe care se decide)
        """
        x = np.array([feature_vector], dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: x})

        prob_abnormal = float(outputs[1][0][1])
        label = "abnormal" if prob_abnormal >= self.threshold else "normal"

        return label, prob_abnormal
