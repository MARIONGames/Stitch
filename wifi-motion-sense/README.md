# wmsense — WiFi motion sensing & human/non-human classification

Detect movement in a space using ordinary WiFi, and tell whether the mover is a
**human** or **not** (pet, fan, machinery) — no cameras, no wearables.

It works by reading the **Channel State Information (CSI)** that a WiFi receiver
already computes for every packet it decodes from your router/AP. When something
moves in the room it perturbs the radio multipath between the router and the
receiver. Those perturbations are measured, turned into a motion signal, and
classified.

```
router (TX) ))) ~~~ person/pet/fan ~~~ ((( receiver (RX, reads CSI)
                     │
                     ▼
        detect motion  ──►  classify mover (human vs non-human)
```

> ⚠️ **Use responsibly.** Sense only spaces you own or have explicit consent to
> monitor. WiFi sensing can reveal presence and activity through walls; treat it
> like any other presence sensor and respect local privacy law.

## Quick start (no hardware needed)

```bash
pip install numpy            # the only hard requirement
# optional: pip install scikit-learn joblib   # enables the RandomForest model

python3 demo.py              # runs every scenario through the full pipeline
```

Expected output:

```
scenario       | motion?   | verdict
------------------------------------------------------------
empty          | no        | no motion detected
human_still    | yes       | HUMAN      (…)
human_moving   | yes       | HUMAN      (…)
pet            | yes       | NON_HUMAN  (…)
fan            | yes       | NON_HUMAN  (…)
```

Live dashboard against the simulator (empty room for 5 s, then a mover enters):

```bash
python3 -m wmsense.app --source sim --scenario human_moving
python3 -m wmsense.app --source sim --scenario fan
```

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Capture | `wmsense/csi.py` | Reads CSI frames (ESP32 serial, pcap CSV, or simulator) and buffers them into uniform windows. |
| Detect | `wmsense/processing.py` | Removes the static channel, band-passes to the motion band (0.1–10 Hz), collapses subcarriers via PCA, and flags motion when energy exceeds an adaptive noise floor. |
| Describe | `wmsense/features.py` | Extracts physically-motivated features (breathing-band power, limb/gait power, mechanical-band power, periodicity, **subcarrier coherence**, …). |
| Classify | `wmsense/classifier.py` | RandomForest (if scikit-learn is installed) or a transparent heuristic fallback → human vs non-human. |

**Why human vs non-human is possible from WiFi alone:**
- A **human body** is a large reflector → its motion is *coherent across many
  subcarriers*, and at rest it still breathes (a slow ~0.15–0.4 Hz signal).
- A **pet** is small → low subcarrier coherence, faster and more irregular.
- A **fan / machine** is strongly periodic in a narrow high band (~4–12 Hz)
  with essentially no breathing component.

The classifier leans on subcarrier coherence (size), the breathing band, and the
mechanical band to separate these.

## Using real hardware

The physics is the same; only the capture backend changes.

### ESP32 (cheapest path)
Flash the [ESP-CSI](https://github.com/espressif/esp-csi) example so the board
prints `CSI_DATA,...` lines over USB serial, then:

```bash
pip install pyserial
python3 -m wmsense.app --source esp32 --port /dev/ttyUSB0
```

### Nexmon CSI (Raspberry Pi / Broadcom) or Intel 5300 CSI Tool
These dump CSI as UDP/pcap. Export each capture to a CSV with a `timestamp`
column followed by either `amp_0,phase_0,…` or `i_0,q_0,…` pairs, then:

```bash
python3 -m wmsense.app --source pcap --file capture.csv --realtime
```

## Training on your own data

The shipped model is trained on the simulator and works out of the box. To train
on real recordings, drop pcap-exported CSVs into a folder, named so each file
starts with its label — `human_livingroom_01.csv`, `non_human_fan_03.csv` — then:

```bash
python3 train.py --csv-dir ./recordings --out model.pkl
python3 -m wmsense.app --source esp32 --model model.pkl
```

`train.py` prints held-out accuracy and a per-recording breakdown.

## Tuning

| Flag | Meaning | Default |
|------|---------|---------|
| `--sensitivity` | Detector threshold, in σ above the noise floor. Lower = more sensitive. | `6.0` |
| `--window` | Analysis window length (seconds). Longer resolves breathing better. | `4.0` |
| `--hop` | Seconds between analyses. | `0.5` |
| `--smoothing` | Windows of majority vote before committing to a label. | `5` |
| `--rate` | Analysis sample rate (Hz). Match it to your CSI packet rate. | `20` |

## Layout

```
wifi-motion-sense/
├── demo.py              # no-hardware end-to-end demo
├── train.py             # train/evaluate the classifier
├── requirements.txt
└── wmsense/
    ├── csi.py           # data model + capture backends
    ├── processing.py    # motion detection
    ├── features.py      # feature extraction
    ├── classifier.py    # human/non-human classifier
    ├── simulator.py     # synthetic CSI generator
    └── app.py           # real-time CLI dashboard
```
