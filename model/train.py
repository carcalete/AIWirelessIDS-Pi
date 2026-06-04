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

# Cele 21 de features calculate de compute_window_features (acelasi set ca
# features/features.py -> extract_features()). Sunt pastrate toate pentru analiza,
# DAR modelul foloseste doar subsetul robust de mai jos.
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

# Features pe care le primeste EFECTIV modelul (vectorul din features_to_vector()).
# Sunt doar features robuste (ratios + rate + diversitate + semnal), NU counts
# absolute de volum (total_packets, *_count, unique_*_macs). Motiv: capturile de
# antrenare si test (si mediul live de pe Pi) au volume de trafic diferite; counts
# absolute invata tipare specifice capturii si produc multe false positives.
# Eliminarea lor a crescut AUC pe test de la 0.83 la 0.89 si a redus FP.
# ATENTIE: aceasta lista trebuie sa fie IDENTICA (nume + ordine) cu feature_order
# din features/features.py -> features_to_vector().
MODEL_FEATURES = [
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

    is_mgmt = ft == 0
    mgmt  = int(is_mgmt.sum())
    ctrl  = int((ft == 1).sum())
    data  = int((ft == 2).sum())

    # Subtype-urile au sens doar in cadrul tipului de cadru: ex. subtype 8 e "beacon"
    # la management, dar "QoS Data" la data. Pipeline-ul live (capture.py) eticheteaza
    # beacon/deauth/probe DOAR pentru cadre management (via layerele scapy dedicate),
    # deci aici gatuim pe is_mgmt ca antrenarea sa fie identica cu inferenta.
    deauth = int((is_mgmt & ((st == 12) | (st == 10))).sum())
    beacon = int((is_mgmt & (st == 8)).sum())
    probe  = int((is_mgmt & ((st == 4) | (st == 5))).sum())

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
# Ferestre de timp (identic cu pipeline-ul live din main.py)
# ---------------------------------------------------------------------------

def assign_time_windows(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    """
    Imparte pachetele in ferestre de timp de `window_seconds`, dupa frame.time_epoch.

    Aceasta e exact semantica din main.py: live, fereastra k contine pachetele cu
    timpul de sosire in [t0 + k*window, t0 + (k+1)*window). Astfel feature-urile
    absolute (total_packets, *_count) sunt pe aceeasi scara la antrenare si live.
    """
    ts = pd.to_numeric(df.iloc[:, COL_TIMESTAMP], errors="coerce").ffill().bfill()
    if ts.isna().all():
        raise ValueError(
            f"Coloana timestamp (index {COL_TIMESTAMP}, frame.time_epoch) e goala/invalida."
        )

    # Avertisment daca timpul nu e monoton crescator: indica un capture concatenat
    # sau un reset de ceas. Ferestrele de timp absolute pot grupa segmente diferite.
    n_back = int((ts.diff() < 0).sum())
    if n_back > 0:
        logger.warning(
            f"  {n_back:,} pachete cu timestamp in scadere (capturi concatenate / reset ceas?). "
            "Ferestrele de timp pot amesteca segmente; verifica datasetul daca rezultatele par ciudate."
        )

    t0 = ts.min()
    return ((ts - t0) // window_seconds).astype("int64")


def window_labels(df: pd.DataFrame, min_attack_packets: int = 1) -> np.ndarray:
    """
    Eticheta binara per fereastra: 1 (abnormal) daca fereastra contine cel putin
    `min_attack_packets` pachete de atac, altfel 0 (normal).

    min_attack_packets=1 -> comportamentul clasic (orice pachet de atac = abnormal).
    Valori mai mari modeleaza un detector de FLOOD: o fereastra cu 1-2 pachete de
    atac inecate in mii de pachete normale e practic imposibil de distins din
    features agregate, deci o tratam ca normala. Mai fidel cu "deauth floods".
    """
    is_attack = (
        df[df.columns[COL_LABEL]].astype(str).str.strip().str.lower().ne("normal")
    )
    counts = is_attack.groupby(df["_win"], sort=True).sum()
    return (counts >= min_attack_packets).astype(int).values


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

    # 2. Grupare in ferestre de TIMP (identic cu main.py / inferenta live)
    win = args.window_seconds
    logger.info(f"Grupare in ferestre de timp de {win}s...")
    df["_win"] = assign_time_windows(df, win)

    n_windows = df["_win"].nunique()
    logger.info(f"  → {n_windows:,} ferestre")

    # 3. Extragere features per fereastra
    logger.info("Extragere features (vectorizat)...")
    X_df = df.groupby("_win", sort=True).apply(compute_window_features)

    # 4. Etichete per fereastra (prag de intensitate: ≥ min_attack_packets)
    y = window_labels(df, args.min_attack_packets)
    logger.info(f"  → eticheta abnormal daca ≥{args.min_attack_packets} pachete de atac/fereastra")

    # Modelul foloseste doar subsetul robust de features (vezi MODEL_FEATURES).
    X = X_df[MODEL_FEATURES].values.astype(np.float32)
    logger.info(f"  → {len(MODEL_FEATURES)} features model: {', '.join(MODEL_FEATURES)}")

    # 4b. Calibrare (domain adaptation): adauga ferestre NORMALE din mediul local.
    # AWID2 e un testbed din 2015; mediul real are alt profil de trafic (ex. multe
    # beacon-uri) pe care modelul il confunda cu impersonation -> false positives.
    # Adaugand trafic normal local ca 'normal' (cu greutate mai mare), modelul invata
    # cum arata "normal" la TINE, fara sa piarda detectia de atac. Randurile sunt deja
    # in formatul celor 12 MODEL_FEATURES (produse de calibrate.py / features_to_vector).
    sample_weight = np.ones(len(X), dtype=np.float32)
    if args.calibration_csv:
        cal = pd.read_csv(args.calibration_csv, header=None).values.astype(np.float32)
        if cal.shape[1] != len(MODEL_FEATURES):
            raise ValueError(
                f"--calibration-csv are {cal.shape[1]} coloane, astept {len(MODEL_FEATURES)} "
                f"(cele 12 MODEL_FEATURES). Regenereaza cu calibrate.py."
            )
        X = np.vstack([X, cal])
        y = np.concatenate([y, np.zeros(len(cal), dtype=y.dtype)])
        sample_weight = np.concatenate([
            sample_weight,
            np.full(len(cal), args.calibration_weight, dtype=np.float32),
        ])
        logger.info(
            f"  → +{len(cal)} ferestre normale de calibrare (mediu local), "
            f"greutate {args.calibration_weight}x/fereastra"
        )

    # 4c. Exemple de ATAC locale: invata modelul semnaturi de atac capturate cu
    # uneltele reale (auth flood, beacon flood, deauth, evil twin) prin
    # `calibrate.py --label attack`. Necesar fiindca AWID2 nu acopera bine toate
    # tipurile, iar calibrarea (normalul) le poate masca pe cele bazate pe beacon.
    if args.attack_csv:
        atk = pd.read_csv(args.attack_csv, header=None).values.astype(np.float32)
        if atk.shape[1] != len(MODEL_FEATURES):
            raise ValueError(
                f"--attack-csv are {atk.shape[1]} coloane, astept {len(MODEL_FEATURES)} "
                f"(cele 12 MODEL_FEATURES). Regenereaza cu calibrate.py --label attack."
            )
        X = np.vstack([X, atk])
        y = np.concatenate([y, np.ones(len(atk), dtype=y.dtype)])
        sample_weight = np.concatenate([
            sample_weight,
            np.full(len(atk), args.attack_weight, dtype=np.float32),
        ])
        logger.info(
            f"  → +{len(atk)} ferestre de ATAC locale, "
            f"greutate {args.attack_weight}x/fereastra"
        )

    n_normal   = int(np.sum(y == 0))
    n_abnormal = int(np.sum(y == 1))
    logger.info(f"  → normal={n_normal:,}  abnormal={n_abnormal:,}  ratio={n_normal/max(n_abnormal,1):.1f}:1")

    if n_abnormal == 0:
        logger.error("Nicio fereastra abnormal detectata. Verifica pozitia coloanei label.")
        return

    # 5. Train/test split (ducem si sample_weight prin split)
    from sklearn.model_selection import train_test_split, cross_val_score
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 6. Antrenare XGBoost
    # Config regularizat: capturile de train si test sunt diferite, deci
    # generalizarea conteaza mai mult decat fit-ul pe train. Arbori mai
    # putin adanci + subsampling + L1/L2 reduc overfitting-ul (gap-ul intre
    # AUC pe train si pe test), ceea ce scade false positives pe captura noua.
    import xgboost as xgb
    logger.info("Antrenare XGBoost (regularizat)...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        scale_pos_weight=n_normal / max(n_abnormal, 1),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    # 7. Evaluare
    from sklearn.metrics import classification_report, roc_auc_score
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['normal', 'abnormal'])}")
    try:
        logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    except Exception:
        pass

    # 7b. Importanta features (gain) — utila pentru analiza din lucrare
    importances = sorted(
        zip(MODEL_FEATURES, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    logger.info("Importanta features (gain):")
    for name, imp in importances:
        logger.info(f"  {name:24s} {imp:.4f}")

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

    # 10. Salveaza lista de features a MODELULUI (ordinea vectorului de intrare)
    names_path = str(output_dir / "feature_names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(MODEL_FEATURES))
    logger.info(f"  → Features salvate: {names_path}")

    logger.info(f"\n✓ Antrenare completa. Modele in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Antrenare model IDS pe features per-fereastra (AWID2)"
    )
    parser.add_argument("--dataset",        required=True,  help="Cale CSV AWID2")
    parser.add_argument("--output",         default="model/", help="Director output modele")
    parser.add_argument("--window-seconds", default=5.0, type=float,
                        help="Durata ferestrei de timp in secunde (default: 5, ca main.py)")
    parser.add_argument("--min-attack-packets", default=1, type=int,
                        help="Prag intensitate: fereastra = atac daca are ≥ N pachete de atac (default: 1)")
    parser.add_argument("--calibration-csv", default=None,
                        help="CSV cu ferestre normale locale (12 coloane, din calibrate.py) adaugate ca 'normal'")
    parser.add_argument("--calibration-weight", default=3.0, type=float,
                        help="Greutatea fiecarei ferestre de calibrare in antrenare (default: 3)")
    parser.add_argument("--attack-csv", default=None,
                        help="CSV cu ferestre de ATAC locale (12 coloane, din calibrate.py --label attack)")
    parser.add_argument("--attack-weight", default=3.0, type=float,
                        help="Greutatea fiecarei ferestre de atac local in antrenare (default: 3)")
    main(parser.parse_args())
