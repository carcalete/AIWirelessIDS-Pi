"""
train.py — Antrenare model IDS pe features per-fereastra de timp (AWID2)
=========================================================================
Genereaza features compatibile cu features/features.py (21 features per window).

Utilizare:
    python train.py --dataset path/to/awid.csv --output model/

Dependente:
    pip install pandas numpy scikit-learn xgboost onnxmltools onnxruntime
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pozitii coloane AWID2 (confirmate experimental din dataset)
# ---------------------------------------------------------------------------
COL_TIMESTAMP  = 3    # frame.time_epoch
COL_FRAME_TYPE = 65   # wlan.fc.type: 0=management, 1=control, 2=data
COL_SUBTYPE    = 66   # wlan.fc.subtype: 8=beacon, 12=deauth, 10=disassoc, 4/5=probe
COL_LEN        = 7    # frame.len
COL_RSSI       = 60   # radiotap.dbm_antsignal
COL_SRC_MAC    = 76   # wlan.ta (transmitter address)
COL_DST_MAC    = 75   # wlan.ra (receiver address)
COL_LABEL      = 154  # class (ultima coloana)

BROADCAST_MAC  = "ff:ff:ff:ff:ff:ff"

# Ordinea exacta a features din features/features.py -> features_to_vector()
FEATURE_NAMES = [
    "total_packets", "deauth_count", "beacon_count", "probe_count",
    "management_count", "control_count", "data_count",
    "unique_src_macs", "unique_dst_macs",
    "avg_packet_length", "avg_inter_arrival_time",
    "deauth_ratio", "beacon_ratio", "probe_ratio",
    "management_ratio", "control_ratio", "data_ratio",
    "avg_rssi", "broadcast_ratio", "top_src_mac_dominance",
    "deauth_to_beacon_ratio",
]


# ---------------------------------------------------------------------------
# Extragere features per fereastra (vectorizat, compatibil cu features.py)
# ---------------------------------------------------------------------------

def compute_window_features(group: pd.DataFrame) -> pd.Series:
    """
    Calculeaza cei 21 features pentru o fereastra de pachete.
    Rezultatul este identic cu features_to_vector() din features/features.py.
    """
    n = len(group)

    ft  = pd.to_numeric(group.iloc[:, COL_FRAME_TYPE], errors="coerce").fillna(-1).astype(int)
    st  = pd.to_numeric(group.iloc[:, COL_SUBTYPE],    errors="coerce").fillna(-1).astype(int)
    lng = pd.to_numeric(group.iloc[:, COL_LEN],        errors="coerce").fillna(0.0)
    rssi_raw = pd.to_numeric(group.iloc[:, COL_RSSI],  errors="coerce").dropna()
    ts_raw   = pd.to_numeric(group.iloc[:, COL_TIMESTAMP], errors="coerce").dropna().sort_values()

    mgmt  = int((ft == 0).sum())
    ctrl  = int((ft == 1).sum())
    data  = int((ft == 2).sum())

    deauth = int(((st == 12) | (st == 10)).sum())
    beacon = int((st == 8).sum())
    probe  = int(((st == 4) | (st == 5)).sum())

    src = group.iloc[:, COL_SRC_MAC].dropna().astype(str).str.lower()
    dst = group.iloc[:, COL_DST_MAC].dropna().astype(str).str.lower()

    broadcast_count = int((dst == BROADCAST_MAC).sum())
    top_src = int(src.value_counts().iloc[0]) if len(src) > 0 else 0

    iats = ts_raw.diff().dropna()

    def r(a, b):
        return float(a) / float(b) if b > 0 else 0.0

    return pd.Series([
        float(n),
        float(deauth), float(beacon), float(probe),
        float(mgmt),   float(ctrl),   float(data),
        float(src.nunique()), float(dst.nunique()),
        float(lng.mean()),
        float(iats.mean()) if len(iats) > 0 else 0.0,
        r(deauth, n), r(beacon, n), r(probe, n),
        r(mgmt, n),   r(ctrl, n),   r(data, n),
        float(rssi_raw.mean()) if len(rssi_raw) > 0 else 0.0,
        r(broadcast_count, n),
        r(top_src, n),
        r(deauth, beacon),
    ], index=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(args):
    np.random.seed(42)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Incarcare dataset
    logger.info(f"Incarcare dataset: {args.dataset}")
    df = pd.read_csv(
        args.dataset,
        header=None,
        na_values=["?", "", " "],
        low_memory=False,
    )
    logger.info(f"  → {df.shape[0]:,} pachete, {df.shape[1]} coloane")

    label_counts = df.iloc[:, COL_LABEL].value_counts()
    logger.info(f"  → Distributie label:\n{label_counts.to_string()}")

    # 2. Grupare secventiala in ferestre de N pachete
    win = args.window_packets
    logger.info(f"Grupare in ferestre de {win} pachete...")
    df["_win"] = np.arange(len(df)) // win

    n_windows = df["_win"].nunique()
    logger.info(f"  → {n_windows:,} ferestre")

    # 3. Extragere features per fereastra
    logger.info("Extragere features (vectorizat)...")
    X_df = df.groupby("_win", sort=True).apply(compute_window_features)

    # 4. Etichete per fereastra: abnormal daca cel putin un pachet e atac
    y = (
        df.groupby("_win")[df.columns[COL_LABEL]]
        .apply(lambda s: int(s.astype(str).str.strip().str.lower().ne("normal").any()))
        .values
    )

    X = X_df.values.astype(np.float32)

    n_normal   = int(np.sum(y == 0))
    n_abnormal = int(np.sum(y == 1))
    logger.info(f"  → normal={n_normal:,}  abnormal={n_abnormal:,}  ratio={n_normal/max(n_abnormal,1):.1f}:1")

    if n_abnormal == 0:
        logger.error("Nicio fereastra abnormal detectata. Verifica pozitia coloanei label.")
        return

    # 5. Train/test split
    from sklearn.model_selection import train_test_split, cross_val_score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 6. Antrenare XGBoost
    import xgboost as xgb
    logger.info("Antrenare XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=n_normal / max(n_abnormal, 1),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 7. Evaluare
    from sklearn.metrics import classification_report, roc_auc_score
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['normal', 'abnormal'])}")
    try:
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    except Exception:
        pass

    # 8. Cross-validare 5-fold
    logger.info("Cross-validare 5-fold...")
    cv = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)
    logger.info(f"  F1 CV: {cv.mean():.4f} ± {cv.std():.4f}")

    # 9. Export ONNX
    logger.info("Export ONNX...")
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = onnxmltools.convert_xgboost(
            model,
            initial_types=[("float_input", FloatTensorType([None, X.shape[1]]))],
        )
        onnx_path = str(output_dir / "ids_xgb.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"  → ONNX salvat: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"  Export ONNX esuat ({e}), salvare pickle fallback.")
        import pickle
        pkl_path = str(output_dir / "ids_xgb.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  → Pickle salvat: {pkl_path}")

    # 10. Salveaza lista de features (folosita in detection.py)
    names_path = str(output_dir / "feature_names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(FEATURE_NAMES))
    logger.info(f"  → Features salvate: {names_path}")

    logger.info(f"\n✓ Antrenare completa. Modele in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Antrenare model IDS pe features per-fereastra (AWID2)"
    )
    parser.add_argument("--dataset",        required=True,  help="Cale CSV AWID2")
    parser.add_argument("--output",         default="model/", help="Director output modele")
    parser.add_argument("--window-packets", default=50, type=int,
                        help="Pachete per fereastra de antrenare (default: 50)")
    main(parser.parse_args())
