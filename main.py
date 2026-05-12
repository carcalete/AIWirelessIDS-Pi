"""
main.py — Orchestrare IDS wireless pe Raspberry Pi
===================================================
Porneste captura, extrage features pe ferestre de timp,
clasifica traficul si actioneaza la detectia unei intruziuni.

Utilizare:
    sudo python main.py --interface wlan0mon

Optiuni complete:
    sudo python main.py --interface wlan0mon --window 5 --threshold 0.75 --block
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ids.log"),
    ],
)
logger = logging.getLogger(__name__)


def main(args):
    from capture.capture import WiFiSniffer
    from features.features import extract_features, features_to_vector
    from detection.detection import Detector
    from response.response import Responder

    logger.info("=" * 50)
    logger.info("  AIWirelessIDS — Pornire sistem")
    logger.info(f"  Interfata : {args.interface}")
    logger.info(f"  Fereastra : {args.window}s")
    logger.info(f"  Prag      : {args.threshold:.0%}")
    logger.info(f"  Blocare   : {'DA' if args.block else 'NU'}")
    logger.info("=" * 50)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(
            f"Modelul ONNX nu a fost gasit: {model_path}\n"
            "Ruleaza mai intai:\n"
            f"  python train.py --dataset <awid.csv> --output model/"
        )
        return

    detector  = Detector(str(model_path))
    responder = Responder(
        threshold=args.threshold,
        log_dir=args.logs,
        block_enabled=args.block,
        interface=args.interface,
    )
    sniffer = WiFiSniffer(interface=args.interface)

    sniffer.start()
    logger.info("Captura pornita. Apasa Ctrl+C pentru oprire.\n")

    windows_processed = 0
    alerts_triggered  = 0

    try:
        while True:
            time.sleep(args.window)
            batch = sniffer.flush()

            if not batch:
                logger.debug("Fereastra goala, astept pachete...")
                continue

            features = extract_features(batch)
            vector   = features_to_vector(features)
            label, confidence = detector.predict(vector)

            windows_processed += 1
            triggered = responder.handle(label, confidence, features, batch)
            if triggered:
                alerts_triggered += 1

            logger.info(
                f"[WIN #{windows_processed:04d}] "
                f"{len(batch):4d} pkt | "
                f"{label:8s} ({confidence:.1%}) | "
                f"deauth={features['deauth_count']:.0f} "
                f"beacon={features['beacon_count']:.0f} "
                f"probe={features['probe_count']:.0f} "
                f"| alerte={alerts_triggered}"
            )

    except KeyboardInterrupt:
        logger.info("\nOprire solicitata...")
    finally:
        sniffer.stop()
        if args.block:
            responder.unblock_all()
        logger.info(
            f"\n=== Sesiune incheiata ==="
            f"\n  Ferestre procesate : {windows_processed}"
            f"\n  Alerte generate    : {alerts_triggered}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Wireless IDS/IPS — Raspberry Pi"
    )
    parser.add_argument(
        "--interface", "-i", default="wlan0mon",
        help="Interfata in monitor mode (default: wlan0mon)",
    )
    parser.add_argument(
        "--model", "-m", default="model/ids_xgb.onnx",
        help="Cale model ONNX (default: model/ids_xgb.onnx)",
    )
    parser.add_argument(
        "--window", "-w", type=int, default=5,
        help="Durata ferestrei de timp in secunde (default: 5)",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.75,
        help="Prag confidence pentru generare alerta (default: 0.75)",
    )
    parser.add_argument(
        "--logs", default="logs/",
        help="Director pentru fisierele de alerta JSONL (default: logs/)",
    )
    parser.add_argument(
        "--block", action="store_true",
        help="Blocheaza MAC-urile suspecte via iptables (necesita root, Linux)",
    )
    main(parser.parse_args())
