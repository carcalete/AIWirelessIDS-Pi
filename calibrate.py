"""
calibrate.py - Captura de trafic NORMAL din mediul local, pentru calibrarea modelului
AWID2 e un testbed din 2015; mediul tau real are alt profil de trafic (ex. multe
beacon-uri de la AP-urile din jur) pe care modelul il poate confunda cu "impersonation"
-> false positives. Acest tool capteaza trafic normal din mediul tau si salveaza cele
12 features per fereastra, ca sa le adaugi ca 'normal' la reantrenare.

RULEAZA DOAR cand NU exista niciun atac in aer (trafic curat)!

Utilizare (pe Pi, ca root):
    sudo python calibrate.py --interface wlan1 --minutes 5 --output model/calib_normal.csv

Apoi pe PC, reantreneaza cu calibrare:
    python model/train.py --dataset <awid_trn> --output model/ \
        --min-attack-packets 5 --calibration-csv model/calib_normal.csv

Output: CSV fara header, 12 coloane = exact MODEL_FEATURES (din features_to_vector()).
"""

import argparse
import csv
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    from capture.capture import WiFiSniffer
    from features.features import extract_features, features_to_vector

    sniffer = WiFiSniffer(interface=args.interface)
    sniffer.start()
    logger.info(
        f"Captura NORMALA pe {args.interface}, {args.minutes} min, fereastra {args.window}s.\n"
        f"!!! NU rula niciun atac acum - trebuie trafic curat !!!"
        if args.label == "normal" else
        f">>> Capturezi exemple de ATAC. Lanseaza atacul ACUM si tine-l pornit <<<"
    )

    is_attack = args.label == "attack"
    rows = []
    skipped = 0
    end = time.time() + args.minutes * 60
    try:
        while time.time() < end:
            time.sleep(args.window)
            batch = sniffer.flush()
            if not batch:
                continue
            feats = extract_features(batch)
            # La label=normal: garda care sare ferestrele ce par atac (multe deauth).
            # La label=attack: capturam TOT (inclusiv deauth-uri).
            if not is_attack and feats.get("deauth_count", 0) > args.max_deauth:
                skipped += 1
                logger.warning(
                    f"  fereastra cu deauth={feats['deauth_count']:.0f} sarita "
                    f"(pare atac, n-o salvez ca normal)"
                )
                continue
            rows.append(features_to_vector(feats))
            logger.info(
                f"  [{len(rows):3d}] fereastra {args.label.upper():7s} "
                f"(pkt={len(batch)}, beacon={feats['beacon_count']:.0f}, "
                f"deauth={feats['deauth_count']:.0f})"
            )
    except KeyboardInterrupt:
        logger.info("Oprire ceruta.")
    finally:
        sniffer.stop()

    if not rows:
        logger.error("Nicio fereastra capturata. Verifica interfata / monitor mode.")
        return

    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    flag = "--attack-csv" if is_attack else "--calibration-csv"
    logger.info(
        f"Salvat {len(rows)} ferestre [{args.label}] in {args.output} "
        f"({skipped} sarite)."
    )
    logger.info(
        f"Reantreneaza pe PC adaugand: {flag} {args.output}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibrare: captura trafic normal local")
    p.add_argument("--interface", "-i", required=True, help="Interfata in monitor mode (ex. wlan1)")
    p.add_argument("--minutes", type=float, default=5.0, help="Durata capturii in minute (default: 5)")
    p.add_argument("--window", "-w", type=int, default=5, help="Fereastra in secunde (default: 5, ca main.py)")
    p.add_argument("--output", "-o", default="model/calib_normal.csv", help="CSV de iesire")
    p.add_argument("--label", choices=["normal", "attack"], default="normal",
                   help="Eticheta capturii: 'normal' (trafic curat) sau 'attack' (lansezi atacul in timpul capturii)")
    p.add_argument("--max-deauth", type=int, default=10,
                   help="La label=normal: sare ferestrele cu mai multe deauth (posibil atac)")
    main(p.parse_args())
