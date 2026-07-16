# mfx100895324 — full beamtime run manifest

Complete run record for mfx100895324 (2 shifts), transcribed from the beamline
run table. This is the full experimental log for reference. The **active
analysis subset** (Shift 1 prepared-mixture calibration series, runs 36–54) is
in `runs.csv`; this file documents the entire beamtime for context.

Detector: `epix100_1` (ePix100), Fe Kα (LiNbO₃ 23-4) + Fe Kβ (Ge 620) on one
chip. Jungfrau used for monitoring/alignment. Fe foil = energy calibration.

Sample chemistry: 2 mM K₃[Fe(CN)₆] (ferricyanide, Fe³⁺) / K₄[Fe(CN)₆]
(ferrocyanide, Fe²⁺) in 1 M NaCl. Echem: 250 µm Pt wire WE, ~800 µm Ag RE.

Legend for `category`:
- `test` / `dark` / `empty` / `water` — setup, background, alignment
- `foil` — Fe foil energy calibration
- `notch`/`focus` — beam/optics scans
- `calib_mix` — prepared-composition standards (Shift 1 analysis set)
- `insitu_echem` — in-situ chronoamperometry (Shift 2), reduction under applied potential

---

## Shift 1 — 2025-06-21 (prepared mixtures)

| Run(s) | Tag | Category | Notes |
|--------|-----|----------|-------|
| 1–6 | test | test | initial tests |
| 7 | DARK | dark | |
| 8–13 | Notch scan | notch | 9870, 9570–9590 step 5 |
| 19 | Fe_Foil | foil | fe foil scan, 1e-3 |
| 20 | DARK | dark | |
| 21–25 | e-3 | foil | Fe foil alignment, kbeta |
| 26 | empty | empty | without beam |
| 27 | empty | empty | e-3, nothing in beampath |
| 28 | water | water | little water, ADE2 |
| 29 | water | water | with water |
| 30 | DARK | dark | |
| 31 | water | water | lead tape on ePix corner; ADE2; dark deployed |
| 32 | Fe(III) before align | calib_mix | Fe(III); OCV 330–350mV |
| 33 | Fe(III) before align | calib_mix | droplet issues; daq crashed |
| 34 | foil | foil | before realigning, e-3 |
| 35 | foil | foil | after kalpha alignment, e-3 |
| **36** | **Fe(III)** | **calib_mix** | **after alignment; reference (0% reduced); 315mV** |
| 37 | Fe(III) | calib_mix | after alignment |
| 38 | Fe(III) | calib_mix | after alignment (short) |
| 39–42 | Fe(II) | calib_mix | ferrocyanide (reduced end-member); 41: 185mV full well |
| 43–46 | Fe(II):Fe(III)_50:50 | calib_mix | 50:50 mix (stopped); 43: 234mV full well |
| **47** | **Fe_Foil** | **foil** | **e-3; metallic foil — EXCLUDED from titration** |
| 48–50 | Fe(II):Fe(III)_20:80 | calib_mix | 20:80 mix; 48: 268mV |
| 51–54 | Fe(II):Fe(III)_10:90 | calib_mix | 10:90 mix; 51: 270mV; reduced tape speed + static gun; 54 very short |
| 55 | DARK | dark | |

**Analysis set = runs 36–54** (calib_mix + foil 47). See `runs.csv`.

---

## Shift 2 — in-situ electrochemistry (chronoamperometry)

Same Fe(III) 2 mM / 1 M NaCl sample, reduced in situ under applied potential.
`CA_<V>` = chronoamperometry at that potential (reducing Fe³⁺ → Fe²⁺).
OCV started ~300mV (lower than Shift 1's 350mV) with a freshly prepared sample.

| Run(s) | Tag | Category | Notes |
|--------|-----|----------|-------|
| 56 | DARK | dark | |
| 57 | ??? | test | unlabeled |
| 58 | foil | foil | e-3 |
| 59–60 | DARK | dark | |
| 61–63 | foil | foil | after alignment, e-3 |
| 64 | water | water | without Jungfrau; loaded fresh Fe(III) |
| 65–67 | Fe(III)_CA_100mV | insitu_echem | +100 mV applied (reducing); 65: 237mV |
| 69 | Fe(III)_after CA | insitu_echem | reference, no external potential; 252mV |
| 70–73 | Fe(III)_after CA | insitu_echem | 71: 273mV |
| 74 | Fe(III)_CA_m100mV | insitu_echem | −100 mV applied; ran out of sample |
| 75–77 | Fe(III)_CA_m100mV | insitu_echem | 75: 298mV; 77: bubble at ~624.9s changed CA |
| 78–80 | water | water | 30 Hz droplet/detector test |
| 81 | empty | empty | 280mV before applying potential |
| 82–86 | Fe(III)_CA_100mV_5ul_per_min | insitu_echem | slower flow rate (5 µL/min); ~4 min / 7200 evt |
| 87–91 | Fe(III)_CA_100mV_PtCVed_1ul_per_min | insitu_echem | roughened Pt (100 CV cycles); 1 µL/min; +100mV |
| 92–99 | Fe(III)_1ulmin | insitu_echem | reference, no potential, low flow; 92: 314mV; 96: OCV jumped 304mV |
| 100 | foil | foil | 5e-4, peak-ratio check, reduced flux |
| 101 | foil | foil | 1e-4, peak-ratio check |
| 106 | Fe_Foil | foil | att 1e-4 |
| 107 | foil | foil | att 1e-3 |
| 108–110 | Fe(III)_CA_100mV_10ulmin_cell3 | insitu_echem | new echem cell (cell3); 10 µL/min; +100mV |
| 111–113 | Fe(III)_CA_100mV_5ulmin_cell3 | insitu_echem | halved flow (5 µL/min) |
| 114 | foil | foil | e-3 (exclude last 10 s) |

---

## Notes for future analysis

- **Shift 2 (65–113)** is the real in-situ reduction dataset — a potential/flow-rate
  series rather than prepared mixtures. When ready, a separate analysis set
  (and `runs_shift2.csv`) should be defined; it can reuse the same pipeline
  YAML and the IAD/LCF deviation tooling with run 36 (or a Shift-2 Fe(III)
  reference like 69/92) as the oxidized reference.
- Fe foil runs (19, 21–25, 34, 35, 47, 58, 61–63, 100, 101, 106, 107, 114) are
  energy-calibration standards — good candidates for refining
  `crystal_detector_distance` in the pipeline YAML.
- Runs 51–54 had reduced tape speed + static gun; run 54 (and several others
  flagged "short") have low shot counts — treat with care.
