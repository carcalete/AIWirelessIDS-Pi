import logging
from typing import Any, Dict, List, Optional

from scapy.all import AsyncSniffer, Packet
from scapy.layers.dot11 import (
    Dot11,
    Dot11AssoReq,
    Dot11AssoResp,
    Dot11Auth,
    Dot11Beacon,
    Dot11Deauth,
    Dot11Disas,
    Dot11Elt,
    Dot11ProbeReq,
    Dot11ProbeResp,
)

logger = logging.getLogger(__name__)

# 802.11 layer -> (frame_type, subtype) mapping

SUBTYPE_MAP = {
    Dot11Beacon:    ("management", "beacon"),
    Dot11Deauth:    ("management", "deauth"),
    Dot11Disas:     ("management", "deauth"),   # treat disassoc as deauth
    Dot11ProbeReq:  ("management", "probe"),
    Dot11ProbeResp: ("management", "probe"),
    Dot11AssoReq:   ("management", "other"),
    Dot11AssoResp:  ("management", "other"),
    Dot11Auth:      ("management", "auth"),     # subtip distinct pt regula auth flood
}

def _get_frame_type_and_subtype(pkt: Packet):
    """
    Determine frame_type and subtype from Scapy layer presence.
    Falls back to Dot11 type field if no known sublayer is found.
    """
    for layer, (frame_type, subtype) in SUBTYPE_MAP.items():
        if pkt.haslayer(layer):
            return frame_type, subtype

    # Fallback: use the raw Dot11 type field
    # type 0 = management, 1 = control, 2 = data
    if pkt.haslayer(Dot11):
        dot11_type = pkt[Dot11].type
        type_map = {0: "management", 1: "control", 2: "data"}
        return type_map.get(dot11_type, "unknown"), "other"

    return "unknown", "other"


# Packet parser

def parse_packet(pkt: Packet) -> Optional[Dict[str, Any]]:
    """
    Parse a Scapy 802.11 packet into a dict compatible with extract_features().
    Returns None if the packet is not a valid 802.11 frame.
    """
    if not pkt.haslayer(Dot11):
        return None

    dot11 = pkt[Dot11]
    frame_type, subtype = _get_frame_type_and_subtype(pkt)

    # RSSI - available only if the interface passes RadioTap headers
    rssi = None
    try:
        from scapy.layers.dot11 import RadioTap
        if pkt.haslayer(RadioTap):
            rssi = pkt[RadioTap].dBm_AntSignal
    except Exception:
        pass

    # Packet length - use raw len (RadioTap included)
    length = len(pkt)

    # SSID + BSSID - doar pentru beacon/probe-resp, folosite la detectia evil twin.
    # BSSID = addr3 (identitatea AP-ului); SSID = primul element Dot11Elt (ID 0).
    ssid = None
    bssid = dot11.addr3 or None
    if pkt.haslayer(Dot11Beacon) or pkt.haslayer(Dot11ProbeResp):
        elt = pkt.getlayer(Dot11Elt)
        while elt is not None:
            if elt.ID == 0:  # 0 = SSID element
                try:
                    ssid = elt.info.decode(errors="ignore")
                except Exception:
                    ssid = None
                break
            elt = elt.payload.getlayer(Dot11Elt)

    return {
        "frame_type": frame_type,
        "subtype":    subtype,
        "src_mac":    dot11.addr2 or None,  # addr2 = transmitter
        "dst_mac":    dot11.addr1 or None,  # addr1 = receiver
        "length":     length,
        "timestamp":  float(pkt.time),
        "rssi":       rssi,
        "ssid":       ssid,                 # numele retelei (doar beacon/probe-resp)
        "bssid":      bssid,                # identitatea AP-ului (addr3)
    }

# Sniffer

class WiFiSniffer:
    """
    Captures 802.11 packets on a monitor-mode interface.

    Usage:
        sniffer = WiFiSniffer(interface="wlan0mon")
        sniffer.start()
        ...
        batch = sniffer.flush()   # grab packets collected so far
        sniffer.stop()
    """

    def __init__(self, interface: str):
        self.interface = interface
        self._buffer: List[Dict[str, Any]] = []
        self._sniffer: Optional[AsyncSniffer] = None

    def _handle_packet(self, pkt: Packet) -> None:
        parsed = parse_packet(pkt)
        if parsed is not None:
            self._buffer.append(parsed)

    def start(self) -> None:
        logger.info(f"Starting capture on {self.interface}")
        self._sniffer = AsyncSniffer(
            iface=self.interface,
            prn=self._handle_packet,
            store=False,        # don't keep raw packets in memory
        )
        self._sniffer.start()

    def stop(self) -> None:
        if self._sniffer:
            self._sniffer.stop()
            logger.info("Capture stopped.")

    def flush(self) -> List[Dict[str, Any]]:
        """
        Return all packets collected since the last flush and clear the buffer.
        Call this once per time window to get the batch for feature extraction.
        """
        batch, self._buffer = self._buffer, []
        return batch
    
# Quick local test  (sudo python3 capture.py)


if __name__ == "__main__":
    import time
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from features.features import extract_features, features_to_vector

    logging.basicConfig(level=logging.INFO)

    INTERFACE = "wlan0mon"  # change to your monitor-mode interface
    WINDOW_SECONDS = 5

    sniffer = WiFiSniffer(interface=INTERFACE)
    sniffer.start()

    try:
        while True:
            time.sleep(WINDOW_SECONDS)
            batch = sniffer.flush()
            print(f"\n--- Window: {len(batch)} packets ---")

            if batch:
                features = extract_features(batch)
                vector = features_to_vector(features)
                print("Features:", features)
                print("Vector:", vector)

    except KeyboardInterrupt:
        sniffer.stop()
        print("Done.")

    '''
    Interface has to be in monitor mode. (Linux) You can set it up with:
    sudo ip link set wlan0 down
    sudo iw wlan0 set monitor none
    sudo ip link set wlan0 up
    '''