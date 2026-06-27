"""evaluate.py - Evaluare model pe un CSV AWID2 extern (ex: test.csv)"""
import argparse, logging, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# train.py se afla in model/ - il adaugam pe path ca evaluate.py sa mearga
# din radacina repo-ului fara PYTHONPATH (`python evaluate.py ...`).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
from train import compute_window_features, assign_time_windows, window_labels, COL_LABEL, MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Incarcare: {args.dataset}")
    df = pd.read_csv(args.dataset, header=None, na_values=["?","","  "], low_memory=False)
    logger.info(f"  -> {df.shape[0]:,} pachete")
    logger.info(f"  -> Distributie label:\n{df.iloc[:, COL_LABEL].value_counts().to_string()}")

    win = args.window_seconds
    df["_win"] = assign_time_windows(df, win)
    logger.info(f"Grupare in ferestre de timp de {win}s -> {df['_win'].nunique():,} ferestre")

    logger.info("Extragere features...")
    X_df = df.groupby("_win", sort=True).apply(compute_window_features)
    y = window_labels(df, args.min_attack_packets)
    logger.info(f"  -> eticheta abnormal daca >={args.min_attack_packets} pachete de atac/fereastra")
    X = X_df[MODEL_FEATURES].values.astype(np.float32)

    n_normal = int(np.sum(y == 0)); n_abnormal = int(np.sum(y == 1))
    logger.info(f"  -> normal={n_normal:,}  abnormal={n_abnormal:,}")

    import onnxruntime as ort
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X})
    y_pred = outputs[0].astype(int)
    y_prob = outputs[1][:, 1]

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
    logger.info(f"\n{'='*50}\nRezultate pe {Path(args.dataset).name}\n{'='*50}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['normal','abnormal'])}")

    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion Matrix (prag implicit 0.5):\n{cm}")
    logger.info(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    logger.info(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    logger.info(f"F1 (weighted): {f1_score(y, y_pred, average='weighted'):.4f}")
    try:
        logger.info(f"ROC-AUC:      {roc_auc_score(y, y_prob):.4f}")
    except Exception:
        pass

    threshold_sweep(y, y_prob, args.target_recall)


def threshold_sweep(y, y_prob, target_recall: float):
    """
    Pentru un IDS conteaza sa NU ratezi atacuri (false negatives mici = recall mare).
    Pragul de decizie pe P(atac) controleaza compromisul: prag mic -> recall mare,
    dar mai multe false positives. Afisam tabelul complet si recomandam pragul.
    """
    y = np.asarray(y); y_prob = np.asarray(y_prob)
    n_pos = int((y == 1).sum())

    logger.info(f"\n{'='*64}\nSweep de prag pe P(atac) - obiectiv: false negatives minim\n{'='*64}")
    logger.info(f"{'prag':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
                f"{'recall':>8} {'precision':>10} {'F1':>7}")

    def metrics_at(t):
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return (t, tp, fp, fn, tn, recall, precision, f1)

    # Tabel afisat la pas 0.05 (lizibil); recomandarile cauta pe grila fina 0.01.
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        _, tp, fp, fn, tn, recall, precision, f1 = metrics_at(t)
        logger.info(f"{t:>6.2f} {tp:>5} {fp:>5} {fn:>5} {tn:>5} "
                    f"{recall:>8.3f} {precision:>10.3f} {f1:>7.3f}")

    rows = [metrics_at(t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]

    # Pentru fiecare tinta de recall, alegem cel mai MARE prag care o atinge
    # (cel mai mare prag => cele mai putine false positives la acel recall).
    logger.info(f"\n--- Praguri recomandate (din {n_pos} ferestre de atac) ---")
    for tgt in (0.90, 0.95, 0.99, 1.00):
        ok = [r for r in rows if r[5] >= tgt]
        if ok:
            t, tp, fp, fn, tn, recall, precision, f1 = max(ok, key=lambda r: r[0])
            logger.info(f"  recall >= {tgt:.2f} -> prag={t:.2f} | "
                        f"recall={recall:.3f} FN={fn} FP={fp} precision={precision:.3f}")
        else:
            logger.info(f"  recall >= {tgt:.2f} -> imposibil de atins cu acest model")

    # Pragul pentru tinta ceruta de utilizator (--target-recall)
    ok = [r for r in rows if r[5] >= target_recall]
    if ok:
        t, tp, fp, fn, tn, recall, precision, f1 = max(ok, key=lambda r: r[0])
        logger.info(
            f"\n>>> Pentru obiectivul tau (recall >= {target_recall:.2f}): "
            f"ruleaza live cu --threshold {t:.2f}"
            f"\n    (recall={recall:.3f}, FN={fn}, FP={fp}, precision={precision:.3f})"
        )
    else:
        logger.info(f"\n>>> recall >= {target_recall:.2f} nu e atins; cel mai bun recall posibil "
                    f"e {max(r[5] for r in rows):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model",   default="model/ids_xgb.onnx")
    p.add_argument("--window-seconds", default=5.0, type=float)
    p.add_argument("--target-recall", default=0.95, type=float,
                   help="Recall tinta pentru recomandarea pragului (default: 0.95)")
    p.add_argument("--min-attack-packets", default=1, type=int,
                   help="Prag intensitate: fereastra = atac daca are >= N pachete de atac (default: 1)")
    main(p.parse_args())
