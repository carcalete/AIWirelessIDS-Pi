#!/usr/bin/env bash
# setup_monitor.sh — pune o interfata Wi-Fi in monitor mode pe un canal dat.
# Reseteaza-se la fiecare reboot, deci rogue-l dupa fiecare pornire.
#
# Utilizare:
#   sudo ./setup_monitor.sh <interfata> <canal>
# Exemple:
#   sudo ./setup_monitor.sh wlan1 6     # pe Pi  (WIDS)
#   sudo ./setup_monitor.sh wlan0 6     # pe Kali (atacator)

set -e

IFACE="${1:-wlan1}"
CH="${2:-6}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Ruleaza cu sudo: sudo $0 $IFACE $CH"; exit 1
fi

echo "[*] Configurez $IFACE -> monitor mode, canal $CH"

# Impiedica NetworkManager sa revina interfata pe 'managed'
nmcli dev set "$IFACE" managed no 2>/dev/null || true

ip link set "$IFACE" down
iw "$IFACE" set monitor control
ip link set "$IFACE" up
iw dev "$IFACE" set channel "$CH"

echo "[*] Stare finala:"
iw dev "$IFACE" info | grep -E "type|channel"
echo "[+] Gata. Daca scrie 'type monitor' si 'channel $CH', esti pregatit."
