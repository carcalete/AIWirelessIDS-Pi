# AIWirelessIDS-Pi

Sistem embedded de **detecție și prevenire a intruziunilor wireless (WIDS/WIPS)**,
care rulează în timp real pe un Raspberry Pi și combină un model de învățare
automată (XGBoost) cu reguli deterministe pentru a detecta atacuri asupra
rețelelor 802.11.

Lucrare de licență — UPB, Facultatea ETTI, Electronică Aplicată.

## Ce detectează

| Atac | Mecanism de detecție |
|------|----------------------|
| Deauthentication / disassociation flood | Model XGBoost |
| Authentication flood | Regulă (nr. cadre auth/fereastră) |
| Beacon flood | Regulă (nr. BSSID-uri unice/fereastră) |
| Evil twin / rogue AP | Regulă (SSID protejat + BSSID în afara whitelist) + izolare activă |

La detecție, sistemul răspunde gradat: **alertă** (consolă + JSONL) →
**containment** evil twin (injecție de deauth către AP-ul fals, protejat prin
whitelist) → **blocare MAC** prin iptables (doar inline).

## Arhitectură

Traficul este capturat pasiv în *monitor mode*, rezumat la fiecare 5 secunde
într-un vector de 12 trăsături robuste și clasificat hibrid:

```
capture  ->  features  ->  detection (model + reguli)  ->  decizie  ->  response
```

| Modul | Fișier | Rol |
|-------|--------|-----|
| Captură | `capture/capture.py` | Sniffer 802.11 (Scapy) pe interfața monitor |
| Trăsături | `features/features.py` | Rezumă fereastra în 12 trăsături numerice |
| Detecție | `detection/detection.py` | Clasificator ONNX (XGBoost) |
| Răspuns | `response/response.py` | Reguli flood/evil-twin + răspuns gradat IPS |
| Orchestrare | `main.py` | Bucla principală în timp real |

## Hardware

- Raspberry Pi (Kali Linux / Debian)
- Adaptor Wi-Fi **Atheros AR9271** (driver `ath9k_htc`) — suportă monitor mode
  și packet injection
- Wi-Fi-ul integrat al Pi-ului rămâne pentru administrare/internet; AR9271 e
  dedicat capturii

## Instalare

```bash
pip install scapy xgboost onnxruntime onnxmltools scikit-learn pandas numpy
```

Setul de date AWID (`model/dataset.csv`, ~870 MB) **nu** este inclus în repo
(depășește limita GitHub); se descarcă separat de pe portalul AWID.

## Utilizare

### 1. Antrenare (offline, pe PC)

```bash
python model/train.py \
    --dataset model/dataset.csv \
    --output model/ \
    --min-attack-packets 5 \
    --calibration-csv model/calib_total.csv \
    --attack-csv model/attacks_local.csv
```

Produce `model/ids_xgb.onnx` (modelul) și `model/feature_names.txt`.

### 2. Evaluare

```bash
python evaluate.py --dataset model/test.csv --model model/ids_xgb.onnx
```

Afișează ROC-AUC, matricea de confuzie și un *sweep* de prag care recomandă
pragul optim de decizie.

### 3. Calibrare la mediu (pe Pi)

Captează câteva minute de trafic **normal** din mediul real, ca să reduci
alarmele false, apoi reantrenează adăugând fișierul cu `--calibration-csv`:

```bash
sudo python calibrate.py --interface wlan1 --minutes 5 --output model/calib_total.csv
```

### 4. Rulare în timp real (pe Pi)

```bash
# pune adaptorul în monitor mode pe un canal
sudo ./scripts/setup_monitor.sh wlan1 6

# pornește sistemul (prag 0.4, protejează un AP legitim împotriva evil twin)
sudo python main.py -i wlan1 -t 0.4 --protect "SSID_LEGITIM:BSSID_LEGITIM"
```

Opțiuni utile:
- `-t, --threshold` — pragul de decizie pe probabilitatea de atac (default 0.75)
- `--protect SSID:BSSID` — AP legitim de protejat (se poate repeta)
- `--beacon-flood-threshold` / `--auth-flood-threshold` — praguri reguli (default 50)
- `--block` — blochează MAC-urile suspecte prin iptables (doar inline, necesită root)

## Unelte de atac pentru testare

Toate testele se fac exclusiv pe rețele și echipamente proprii, într-un mediu
controlat:

```bash
sudo aireplay-ng --deauth 0 -a <BSSID> wlan0      # deauth flood
sudo mdk4 wlan0 a -a <BSSID>                       # auth flood
sudo mdk4 wlan0 b -c 6                              # beacon flood
sudo airbase-ng -e <SSID> -c 6 wlan0               # evil twin
```

## Structura proiectului

```
.
├── main.py                 # bucla principală în timp real
├── calibrate.py            # captură trafic local pentru calibrare
├── evaluate.py             # evaluare model pe set de test (ROC, confuzie, sweep)
├── capture/capture.py      # sniffer 802.11
├── features/features.py    # extragere trăsături
├── detection/detection.py  # clasificator ONNX
├── response/response.py    # reguli + răspuns gradat IPS
├── model/
│   ├── train.py            # antrenare XGBoost -> ONNX
│   ├── ids_xgb.onnx        # modelul antrenat
│   └── feature_names.txt   # ordinea trăsăturilor modelului
└── scripts/setup_monitor.sh
```
