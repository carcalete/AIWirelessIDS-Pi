from collections import Counter
from statistics import mean
from typing import Any, Dict, List


def safe_mean(values: List[float]) -> float:
    """Return the mean of a list, or 0.0 if the list is empty."""
    return mean(values) if values else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    """Avoid division by zero when computing ratios."""
    return numerator / denominator if denominator else 0.0


def extract_features(packet_batch: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract numerical features from a batch (time window) of packets.

    Expected packet format for now (simulated / normalized):
    {
        "frame_type": "management" | "control" | "data",
        "subtype": "beacon" | "deauth" | "probe" | "other",
        "src_mac": "AA:BB:CC:DD:EE:FF",
        "dst_mac": "11:22:33:44:55:66",
        "length": 128,
        "timestamp": 1712345678.123,
        "rssi": -48   # optional
    }

    Returns:
        Dictionary of extracted numerical features.
    """

    total_packets = len(packet_batch)

    if total_packets == 0: # Handle empty batch case to avoid division by zero and other issues
        return {
            "total_packets": 0.0,
            "deauth_count": 0.0,
            "beacon_count": 0.0,
            "probe_count": 0.0,
            "management_count": 0.0,
            "control_count": 0.0,
            "data_count": 0.0,
            "unique_src_macs": 0.0,
            "unique_dst_macs": 0.0,
            "avg_packet_length": 0.0,
            "avg_inter_arrival_time": 0.0,
            "deauth_ratio": 0.0,
            "beacon_ratio": 0.0,
            "probe_ratio": 0.0,
            "management_ratio": 0.0,
            "control_ratio": 0.0,
            "data_ratio": 0.0,
            "avg_rssi": 0.0,
            "broadcast_ratio": 0.0,
            "top_src_mac_dominance": 0.0,
            "deauth_to_beacon_ratio": 0.0,
        }

    frame_types = [pkt.get("frame_type", "unknown") for pkt in packet_batch]
    subtypes = [pkt.get("subtype", "other") for pkt in packet_batch]
    src_macs = [pkt.get("src_mac") for pkt in packet_batch if pkt.get("src_mac")]
    dst_macs = [pkt.get("dst_mac") for pkt in packet_batch if pkt.get("dst_mac")]

    lengths = []
    for pkt in packet_batch:
        try:
            lengths.append(float(pkt.get("length", 0)))
        except (ValueError, TypeError):
            lengths.append(0.0) # Default to 0 if length is missing or invalid

    rssis = []
    for pkt in packet_batch:
        if pkt.get("rssi") is not None:
            try:
                rssis.append(float(pkt["rssi"]))
            except (ValueError, TypeError):
                pass # Ignore invalid RSSI values

    type_counter = Counter(frame_types)
    subtype_counter = Counter(subtypes)

    # Sort timestamps to compute inter-arrival times
    timestamps = [
        float(pkt["timestamp"])
        for pkt in packet_batch
        if pkt.get("timestamp") is not None
    ]

    inter_arrival_times = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    deauth_count = float(subtype_counter.get("deauth", 0))
    beacon_count = float(subtype_counter.get("beacon", 0))
    probe_count = float(subtype_counter.get("probe", 0))

    management_count = float(type_counter.get("management", 0))
    control_count = float(type_counter.get("control", 0))
    data_count = float(type_counter.get("data", 0))

    features = {
        "total_packets": float(total_packets),

        # Attack-relevant counts
        "deauth_count": deauth_count,
        "beacon_count": beacon_count,
        "probe_count": probe_count,

        # Basic frame distribution
        "management_count": management_count,
        "control_count": control_count,
        "data_count": data_count,

        # Diversity/behavior
        "unique_src_macs": float(len(set(src_macs))),
        "unique_dst_macs": float(len(set(dst_macs))),

        # Packet characteristics
        "avg_packet_length": safe_mean(lengths),
        "avg_inter_arrival_time": safe_mean(inter_arrival_times),
        "avg_rssi": safe_mean(rssis),

        # Ratios
        "deauth_ratio": safe_ratio(deauth_count, total_packets),
        "beacon_ratio": safe_ratio(beacon_count, total_packets),
        "probe_ratio": safe_ratio(probe_count, total_packets),
        "management_ratio": safe_ratio(management_count, total_packets),
        "control_ratio": safe_ratio(control_count, total_packets),
        "data_ratio": safe_ratio(data_count, total_packets),
    }

    broadcast_mac = "FF:FF:FF:FF:FF:FF"
    broadcast_count = float(sum(1 for mac in dst_macs if mac == broadcast_mac))
    top_src_count = float(Counter(src_macs).most_common(1)[0][1]) if src_macs else 0.0

    features["broadcast_ratio"] = safe_ratio(broadcast_count, total_packets)
    features["top_src_mac_dominance"] = safe_ratio(top_src_count, total_packets)
    features["deauth_to_beacon_ratio"] = safe_ratio(deauth_count, beacon_count)

    return features


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """
    Convert a feature dictionary into a model-ready ordered list.
    Keep this order stable across training and inference.
    """
    feature_order = [
        "total_packets",
        "deauth_count",
        "beacon_count",
        "probe_count",
        "management_count",
        "control_count",
        "data_count",
        "unique_src_macs",
        "unique_dst_macs",
        "avg_packet_length",
        "avg_inter_arrival_time",
        "deauth_ratio",
        "beacon_ratio",
        "probe_ratio",
        "management_ratio",
        "control_ratio",
        "data_ratio",
        "avg_rssi",
        "broadcast_ratio",
        "top_src_mac_dominance",
        "deauth_to_beacon_ratio",
    ]

    return [features[name] for name in feature_order]


if __name__ == "__main__":
    # Quick local test with fake packets
    sample_packets = [
        {
            "frame_type": "management",
            "subtype": "deauth",
            "src_mac": "AA:AA:AA:AA:AA:01",
            "dst_mac": "FF:FF:FF:FF:FF:FF",
            "length": 64,
            "timestamp": 1.00,
            "rssi": -45,
        },
        {
            "frame_type": "management",
            "subtype": "deauth",
            "src_mac": "AA:AA:AA:AA:AA:01",
            "dst_mac": "FF:FF:FF:FF:FF:FF",
            "length": 64,
            "timestamp": 1.05,
            "rssi": -46,
        },
        {
            "frame_type": "management",
            "subtype": "beacon",
            "src_mac": "BB:BB:BB:BB:BB:02",
            "dst_mac": "FF:FF:FF:FF:FF:FF",
            "length": 128,
            "timestamp": 1.10,
            "rssi": -60,
        },
        {
            "frame_type": "data",
            "subtype": "other",
            "src_mac": "CC:CC:CC:CC:CC:03",
            "dst_mac": "DD:DD:DD:DD:DD:04",
            "length": 512,
            "timestamp": 1.30,
            "rssi": -52,
        },
    ]

    extracted = extract_features(sample_packets)
    vector = features_to_vector(extracted)

    print("Extracted features:")
    for key, value in extracted.items():
        print(f"{key}: {value}")

    print("\nFeature vector:")
    print(vector)