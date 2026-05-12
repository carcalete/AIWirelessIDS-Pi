"""
response.py — Modul de raspuns la intruziuni detectate.

Actiuni disponibile:
  - Logare alerta (intotdeauna activa) in consola si fisier JSONL zilnic
  - Blocare MAC via iptables (Linux/Pi, necesita root, --block)
"""

import json
import logging
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Responder:
    """
    Interpreteaza rezultatul detectiei si actioneaza corespunzator.

    Usage:
        responder = Responder(threshold=0.75, log_dir="logs/")
        triggered = responder.handle(label, confidence, features, packet_batch)
    """

    def __init__(
        self,
        threshold: float = 0.75,
        log_dir: Optional[str] = "logs/",
        block_enabled: bool = False,
        interface: Optional[str] = None,
    ):
        self.threshold     = threshold
        self.block_enabled = block_enabled
        self.interface     = interface
        self._blocked: set = set()

        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # API public
    # ------------------------------------------------------------------

    def handle(
        self,
        label: str,
        confidence: float,
        features: Dict[str, float],
        packet_batch: list,
    ) -> bool:
        """
        Proceseaza rezultatul unei ferestre de detecție.

        Returns:
            True daca a fost detectata o intruziune si s-a actionat.
        """
        if label == "normal" or confidence < self.threshold:
            return False

        suspects = self._find_suspects(packet_batch)
        self._log_alert(confidence, features, suspects)

        if self.block_enabled:
            for mac in suspects:
                self._block_mac(mac)

        return True

    def unblock_all(self):
        """Deblocheaza toate MAC-urile la oprire (cleanup)."""
        for mac in list(self._blocked):
            try:
                subprocess.run(
                    ["iptables", "-D", "INPUT", "-m", "mac",
                     "--mac-source", mac, "-j", "DROP"],
                    check=True, capture_output=True,
                )
                logger.info(f"[UNBLOCK] {mac}")
            except Exception:
                pass
        self._blocked.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_suspects(self, packet_batch: list) -> List[str]:
        """
        Returneaza MAC-urile care domina fereastra (> 30% din pachete).
        Ignora broadcast si None.
        """
        src_macs = [
            p["src_mac"].lower()
            for p in packet_batch
            if p.get("src_mac") and p["src_mac"].lower() != "ff:ff:ff:ff:ff:ff"
        ]
        if not src_macs:
            return []
        total = len(src_macs)
        return [
            mac for mac, cnt in Counter(src_macs).most_common(3)
            if cnt / total > 0.30
        ]

    def _log_alert(
        self,
        confidence: float,
        features: Dict[str, float],
        suspects: List[str],
    ):
        ts = datetime.now()
        logger.warning(
            f"[ALERT] Intruziune detectata | "
            f"confidence={confidence:.1%} | "
            f"deauth={features.get('deauth_count', 0):.0f} | "
            f"beacon={features.get('beacon_count', 0):.0f} | "
            f"probe={features.get('probe_count', 0):.0f} | "
            f"suspects={suspects}"
        )

        if not self._log_dir:
            return

        entry = {
            "timestamp":   ts.isoformat(),
            "confidence":  round(confidence, 4),
            "suspect_macs": suspects,
            "features": {k: round(v, 4) for k, v in features.items()},
        }
        log_file = self._log_dir / f"alerts_{ts.strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _block_mac(self, mac: str):
        """Blocheaza un MAC via iptables (necesita root pe Linux)."""
        if mac in self._blocked:
            return
        self._blocked.add(mac)
        try:
            subprocess.run(
                ["iptables", "-A", "INPUT", "-m", "mac",
                 "--mac-source", mac, "-j", "DROP"],
                check=True, capture_output=True,
            )
            logger.warning(f"[BLOCK] MAC blocat: {mac}")
        except FileNotFoundError:
            logger.error("[BLOCK] iptables negasit (necesita Linux + root).")
        except subprocess.CalledProcessError as e:
            logger.error(f"[BLOCK] Esec blocare {mac}: {e.stderr.decode()}")
