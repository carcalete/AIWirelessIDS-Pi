"""
response.py - Modul de raspuns la intruziuni detectate (raspuns gradat).

Niveluri de raspuns:
  - Nivel 1 ALERTARE (mereu): consola + fisier JSONL zilnic.
  - Nivel 2 CONTAINMENT evil twin / rogue AP: injecteaza deauth catre BSSID-ul fals
    (clientii pleaca de pe AP-ul atacatorului). SIGUR prin whitelist (--protect):
    nu atinge niciodata un AP legitim. Functioneaza fiindca atacatorul are un BSSID
    REAL, targetabil (spre deosebire de deauth flood, care e spoofed -> doar alerta).
  - Nivel 3 BLOCARE MAC inline via iptables (--block): real doar daca Pi e gateway/AP.
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
        responder = Responder(threshold=0.75, log_dir="logs/", protect=["gilbert:aa:bb:..."])
        responder.check_rogue_aps(packet_batch)            # containment evil twin
        triggered = responder.handle(label, confidence, features, packet_batch)
    """

    def __init__(
        self,
        threshold: float = 0.75,
        log_dir: Optional[str] = "logs/",
        block_enabled: bool = False,
        interface: Optional[str] = None,
        protect: Optional[List[str]] = None,
        beacon_flood_threshold: int = 50,
        auth_flood_threshold: int = 50,
    ):
        self.threshold     = threshold
        self.block_enabled = block_enabled
        self.interface     = interface
        self.beacon_flood_threshold = beacon_flood_threshold
        self.auth_flood_threshold   = auth_flood_threshold
        self._blocked: set = set()
        self._contained: set = set()

        # Whitelist AP-uri legitime: { ssid_lower: {bssid_lower, ...} }.
        # Format intrare: "SSID:BSSID" (ex. "gilbert:4a:7a:35:f4:da:71").
        # Containment-ul se face DOAR pentru SSID-uri protejate cu BSSID NEcunoscut.
        self._protect: Dict[str, set] = {}
        for entry in (protect or []):
            if ":" not in entry:
                logger.warning(f"--protect ignora '{entry}' (format: SSID:BSSID)")
                continue
            ssid, bssid = entry.split(":", 1)
            self._protect.setdefault(ssid.lower(), set()).add(bssid.lower())
        if self._protect:
            logger.info(f"Containment evil twin activ pentru: {dict((k, list(v)) for k,v in self._protect.items())}")

        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    # Nivel 2 - Containment evil twin / rogue AP

    def check_rogue_aps(self, packet_batch: list) -> List[str]:
        """
        Detecteaza evil twin (regula): un beacon cu un SSID PROTEJAT dar un BSSID care
        NU e cel legitim -> e un AP fals care imita reteaua ta. Il contine prin deauth.
        Niciodata nu atinge un BSSID din whitelist. Ruleaza la fiecare fereastra.
        """
        if not self._protect:
            return []
        contained = []
        for pkt in packet_batch:
            ssid = (pkt.get("ssid") or "").lower()
            bssid = (pkt.get("bssid") or "").lower()
            if not ssid or ssid not in self._protect or not bssid:
                continue
            legit = self._protect[ssid]
            if bssid in legit:
                continue                       # AP legitim -> niciodata atins
            if bssid in self._contained:
                continue                       # deja tratat
            self._contain_rogue(bssid, ssid)
            contained.append(bssid)
        return contained

    def _contain_rogue(self, bssid: str, ssid: str):
        """
        Trimite cadre deauth catre BSSID-ul rogue (broadcast, spoofing BSSID-ul fals)
        -> clientii conectati la evil twin se deconecteaza de pe el.
        """
        self._contained.add(bssid)
        logger.warning(f"[CONTAIN] Evil twin '{ssid}' BSSID={bssid} - trimit deauth de containment")
        if not self.interface:
            logger.error("[CONTAIN] interfata lipseste, nu pot injecta.")
            return
        try:
            from scapy.all import RadioTap, sendp
            from scapy.layers.dot11 import Dot11, Dot11Deauth
            pkt = (RadioTap() /
                   Dot11(addr1="ff:ff:ff:ff:ff:ff", addr2=bssid, addr3=bssid) /
                   Dot11Deauth(reason=7))
            sendp(pkt, iface=self.interface, count=10, inter=0.1, verbose=False)
            logger.warning(f"[CONTAIN] {bssid} - 10 cadre deauth trimise.")
        except Exception as e:
            logger.error(f"[CONTAIN] Esec injectare catre {bssid}: {e}")

    # Detectie pe REGULI (atacuri pe care AI-ul nu le poate prinde fara FP)

    def check_flood_rules(self, packet_batch: list) -> List[str]:
        """
        Detectie pe reguli pentru atacuri cu semnatura neambigua, independenta de mediu:
          - Beacon flood: prea multe BSSID-uri UNICE (AP-uri false) intr-o fereastra.
            Un mediu normal are zeci de AP-uri; un flood creeaza sute de MAC-uri random.
          - Auth flood: prea multe cadre de autentificare intr-o fereastra (DoS pe AP).
        Aceste semnale se suprapun cu trafic normal beacon-heavy in spatiul ML (-> FP),
        de aceea sunt tratate cu praguri pe numere brute, nu de model.
        Sunt spoofed / MAC-uri random -> raspuns = ALERTA (nu containment).
        """
        detected = []

        beacon_bssids = {
            (p.get("bssid") or "").lower()
            for p in packet_batch
            if p.get("subtype") == "beacon" and p.get("bssid")
        }
        if len(beacon_bssids) >= self.beacon_flood_threshold:
            logger.warning(
                f"[ALERT] Beacon flood detectat (regula) | "
                f"{len(beacon_bssids)} BSSID-uri unice >= prag {self.beacon_flood_threshold}"
            )
            self._log_rule_alert("beacon_flood", {"unique_bssids": len(beacon_bssids)})
            detected.append("beacon_flood")

        auth_count = sum(1 for p in packet_batch if p.get("subtype") == "auth")
        if auth_count >= self.auth_flood_threshold:
            logger.warning(
                f"[ALERT] Auth flood detectat (regula) | "
                f"{auth_count} cadre auth >= prag {self.auth_flood_threshold}"
            )
            self._log_rule_alert("auth_flood", {"auth_frames": auth_count})
            detected.append("auth_flood")

        return detected

    def _log_rule_alert(self, attack: str, details: dict):
        if not self._log_dir:
            return
        ts = datetime.now()
        entry = {"timestamp": ts.isoformat(), "detector": "rule",
                 "attack": attack, "details": details}
        log_file = self._log_dir / f"alerts_{ts.strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # API public

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

    # Internals

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
