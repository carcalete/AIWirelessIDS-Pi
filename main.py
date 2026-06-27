"""
Starts packet capture, extracts features over time windows,
classifies traffic and takes action on intrusion detection.

Usage:
    sudo python main.py --interface wlan0mon

Complete options:
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
    logger.info("  AIWirelessIDS - Starting system")
    logger.info(f"  Interface : {args.interface}")
    logger.info(f"  Window    : {args.window}s")
    logger.info(f"  Threshold : {args.threshold:.0%}")
    logger.info(f"  Blocking  : {'YES' if args.block else 'NO'}")
    logger.info("=" * 50)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(
            f"Model ONNX not found: {model_path}\n"
            "Run the following command first:\n"
            f"  python train.py --dataset <awid.csv> --output model/"
        )
        return

    detector  = Detector(str(model_path), threshold=args.threshold)
    responder = Responder(
        threshold=args.threshold,
        log_dir=args.logs,
        block_enabled=args.block,
        interface=args.interface,
        protect=args.protect,
        beacon_flood_threshold=args.beacon_flood_threshold,
        auth_flood_threshold=args.auth_flood_threshold,
    )
    sniffer = WiFiSniffer(interface=args.interface)

    sniffer.start()
    logger.info("Capture started. Press Ctrl+C to stop.\n")

    windows_processed = 0
    alerts_triggered  = 0

    try:
        while True:
            time.sleep(args.window)
            batch = sniffer.flush()

            if not batch:
                logger.debug("Empty window, waiting for packets...")
                continue

            features = extract_features(batch)
            vector   = features_to_vector(features)
            label, confidence = detector.predict(vector)

            windows_processed += 1
            # Detectie pe REGULI (independenta de mediu): beacon/auth flood + evil twin
            responder.check_rogue_aps(batch)              # evil twin -> containment
            rule_hits = responder.check_flood_rules(batch)  # beacon/auth flood -> alerta
            # Detectie AI: deauth/flooding -> alerta + (block inline)
            triggered = responder.handle(label, confidence, features, batch)
            if triggered or rule_hits:
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
        logger.info("\nStop requested...")
    finally:
        sniffer.stop()
        if args.block:
            responder.unblock_all()
        logger.info(
            f"\n=== Session ended ==="
            f"\n  Windows processed : {windows_processed}"
            f"\n  Alerts generated    : {alerts_triggered}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Wireless IDS/IPS - Raspberry Pi"
    )
    parser.add_argument(
        "--interface", "-i", default="wlan0mon",
        help="Interface in monitor mode (default: wlan0mon)",
    )
    parser.add_argument(
        "--model", "-m", default="model/ids_xgb.onnx",
        help="Path to ONNX model (default: model/ids_xgb.onnx)",
    )
    parser.add_argument(
        "--window", "-w", type=int, default=5,
        help="Window duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.75,
        help="Confidence threshold for alert generation (default: 0.75)",
    )
    parser.add_argument(
        "--logs", default="logs/",
        help="Directory for alert JSONL files (default: logs/)",
    )
    parser.add_argument(
        "--block", action="store_true",
        help="Block suspicious MAC addresses via iptables (requires root, Linux)",
    )
    parser.add_argument(
        "--protect", action="append", default=[], metavar="SSID:BSSID",
        help="AP legitim de protejat (whitelist). Orice alt BSSID pe acest SSID = evil twin "
             "-> containment prin deauth. Se poate repeta. Ex: --protect gilbert:4a:7a:35:f4:da:71",
    )
    parser.add_argument(
        "--beacon-flood-threshold", type=int, default=50,
        help="Beacon flood: nr minim de BSSID-uri unice/fereastra (default: 50)",
    )
    parser.add_argument(
        "--auth-flood-threshold", type=int, default=50,
        help="Auth flood: nr minim de cadre auth/fereastra (default: 50)",
    )
    main(parser.parse_args())
