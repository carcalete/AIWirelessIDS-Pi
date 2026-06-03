"""
replay.py - Testeaza pipeline-ul IDS pe un CSV AWID2 (fara sniffer).

Citeste CSV-ul, il sparge in ferestre de N pachete, ruleaza acelasi flow
ca main.py (extract_features -> Detector -> Responder) si afiseaza la final
metrici (accuracy, precision, recall, F1, confusion matrix) pe ferestre.

Usage:
    python replay.py --dataset test.csv --model model/ids_xgb.onnx
    python replay.py --dataset test.csv --window-packets 50 --threshold 0.75
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from features.features import extract_features, features_to_vector
from detection.detection import Detector
from response.response import Responder

# Aceleasi pozitii de coloane ca in train.py (AWID2)
COL_TIMESTAMP  = 3
COL_FRAME_TYPE = 65
COL_SUBTYPE    = 66
COL_LEN        = 7
COL_RSSI       = 60
COL_SRC_MAC    = 76
COL_DST_MAC    = 75
COL_LABEL      = 154

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _ftype_to_str(code):
    return {0: "management", 1: "control", 2: "data"}.get(code, "unknown")


def _subtype_to_str(code):
    if code in (12, 10): return "deauth"
    if code == 8:        return "beacon"
    if code in (4, 5):   return "probe"
    return "other"


def _row_to_packet(row) -> dict:
    """Converteste un rand AWID2 in dict-ul asteptat de features.extract_features."""
    def _safe_int(v):
        try: return int(float(v))
        except (ValueError, TypeError): return -1

    def _safe_float(v):
        try: return float(v)
        except (ValueError, TypeError): return None

    def _mac(v):
        if v is None: return None
        s = str(v).strip()
        return None if s in ("?", "", "nan") else s

    return {
        "frame_type": _ftype_to_str(_safe_int(row[COL_FRAME_TYPE])),
        "subtype":    _subtype_to_str(_safe_int(row[COL_SUBTYPE])),
        "src_mac":    _mac(row[COL_SRC_MAC]),
        "dst_mac":    _mac(row[COL_DST_MAC]),
        "length":     _safe_float(row[COL_LEN]) or 0.0,
        "timestamp":  _safe_float(row[COL_TIMESTAMP]),
        "rssi":       _safe_float(row[COL_RSSI]),
    }


def main(args):
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model nu exista: {model_path}")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset nu exista: {dataset_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  AIWirelessIDS - REPLAY mode")
    logger.info(f"  Dataset   : {dataset_path}")
    logger.info(f"  Model     : {model_path}")
    logger.info(f"  Window    : {args.window_packets} pachete")
    logger.info(f"  Threshold : {args.threshold:.0%}")
    logger.info("=" * 60)

    # 1. Init pipeline (identic cu main.py)
    detector  = Detector(str(model_path))
    responder = Responder(
        threshold=args.threshold,
        log_dir=args.logs,
        block_enabled=False,   # niciodata blocat in replay
        interface=None,
    )

    # 2. Incarca CSV
    logger.info("Incarcare CSV...")
    df = pd.read_csv(dataset_path, header=None, na_values=["?", "", " "], low_memory=False)
    logger.info(f"  -> {len(df):,} pachete, {df.shape[1]} coloane")

    n = len(df)
    win = args.window_packets
    n_windows = n // win
    logger.info(f"  -> {n_windows:,} ferestre complete de {win} pachete\n")

    # 3. Loop pe ferestre (echivalent cu while True din main.py)
    rows = df.values.tolist()  # mai rapid decat iterrows
    label_col = df.iloc[:, COL_LABEL].astype(str).str.strip().str.lower().values

    windows_processed = 0
    alerts_triggered  = 0
    tp = fp = tn = fn = 0  # confusion matrix la nivel de fereastra

    for w in range(n_windows):
        start = w * win
        end   = start + win
        batch = [_row_to_packet(r) for r in rows[start:end]]

        # Eticheta reala = orice pachet din fereastra e atac
        true_abnormal = bool((label_col[start:end] != "normal").any())

        # Acelasi flow ca main.py
        feats = extract_features(batch)
        vec   = features_to_vector(feats)
        label, conf = detector.predict(vec)

        windows_processed += 1
        triggered = responder.handle(label, conf, feats, batch)
        if triggered:
            alerts_triggered += 1

        # Update confusion matrix
        pred_abnormal = (label == "abnormal" and conf >= args.threshold)
        if   true_abnormal and pred_abnormal:     tp += 1
        elif true_abnormal and not pred_abnormal: fn += 1
        elif not true_abnormal and pred_abnormal: fp += 1
        else:                                     tn += 1

        # Log scurt per fereastra (doar daca verbose, ca sa nu spameze)
        if args.verbose or triggered:
            marker = "!" if triggered else " "
            true_marker = "A" if true_abnormal else "N"
            logger.info(
                f"[WIN #{windows_processed:04d}]{marker} true={true_marker} "
                f"pred={label:8s} ({conf:.1%}) "
                f"deauth={feats['deauth_count']:.0f} "
                f"beacon={feats['beacon_count']:.0f} "
                f"probe={feats['probe_count']:.0f}"
            )

    # 4. Raport final
    print()
    print("=" * 60)
    print("  REZULTATE")
    print("=" * 60)
    print(f"  Ferestre procesate    : {windows_processed:,}")
    print(f"  Alerte declansate     : {alerts_triggered:,}")
    print()
    print("  Confusion matrix (la nivel de fereastra):")
    print(f"                  pred normal   pred abnormal")
    print(f"   true normal    {tn:>10}   {fp:>10}")
    print(f"   true abnormal  {fn:>10}   {tp:>10}")
    print()

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (din alertele declansate, % corecte)")
    print(f"  Recall    : {recall:.4f}  (din atacurile reale, % detectate)")
    print(f"  F1-score  : {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay AWID2 CSV prin pipeline-ul IDS (testare offline)"
    )
    parser.add_argument("--dataset",        required=True, help="Cale CSV AWID2")
    parser.add_argument("--model", "-m",    default="model/ids_xgb.onnx",
                        help="Model ONNX (default: model/ids_xgb.onnx)")
    parser.add_argument("--window-packets", type=int, default=50,
                        help="Pachete per fereastra (default: 50, IDENTIC cu antrenarea)")
    parser.add_argument("--threshold", "-t", type=float, default=0.75,
                        help="Prag confidence pentru alerta (default: 0.75)")
    parser.add_argument("--logs",           default="logs/",
                        help="Director pentru fisierele de alerte (default: logs/)")
    parser.add_argument("--verbose", "-v",  action="store_true",
                        help="Afiseaza toate ferestrele, nu doar alertele")
    main(parser.parse_args())