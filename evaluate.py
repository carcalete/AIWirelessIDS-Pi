"""evaluate.py — Evaluare model pe un CSV AWID2 extern (ex: test.csv)"""
import argparse, logging, sys
import numpy as np
import pandas as pd
from pathlib import Path
from train import compute_window_features, COL_LABEL, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Evaluating model on: {args.dataset}")
    df = pd.read_csv(args.dataset, header=None, na_values=["?","","  "], low_memory=False)
    logger.info(f"  → {df.shape[0]:,} packets")
    logger.info(f"  → Label distribution:\n{df.iloc[:, COL_LABEL].value_counts().to_string()}")

    win = args.window_packets
    df["_win"] = np.arange(len(df)) // win
    logger.info(f"Grouping into windows of {win} packets → {df['_win'].nunique():,} windows")

    logger.info("Extracting features...")
    X_df = df.groupby("_win", sort=True).apply(compute_window_features)
    y = (
        df.groupby("_win")[df.columns[COL_LABEL]]
        .apply(lambda s: int(s.astype(str).str.strip().str.lower().ne("normal").any()))
        .values
    )
    X = X_df.values.astype(np.float32)

    n_normal = int(np.sum(y == 0)); n_abnormal = int(np.sum(y == 1))
    logger.info(f"  → normal={n_normal:,}  abnormal={n_abnormal:,}")

    import onnxruntime as ort
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X})
    y_pred = outputs[0].astype(int)
    y_prob = outputs[1][:, 1]

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
    logger.info(f"\n{'='*50}\nResults on {Path(args.dataset).name}\n{'='*50}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['normal','abnormal'])}")

    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    logger.info(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    logger.info(f"F1 (weighted): {f1_score(y, y_pred, average='weighted'):.4f}")
    try:
        logger.info(f"ROC-AUC:      {roc_auc_score(y, y_prob):.4f}")
    except Exception:
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model",   default="model/ids_xgb.onnx")
    p.add_argument("--window-packets", default=50, type=int)
    main(p.parse_args())
