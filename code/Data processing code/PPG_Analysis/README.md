# EarSense offline PPG analysis

This cleaned project contains only offline analysis of three-channel PPG data.
Data streaming, hardware/Arduino interfaces, real-time plots, questionnaires,
IDE metadata, caches, and example EEG data have been removed.

## Input

An `.xlsx` workbook containing these columns on every analyzed sheet:

- `Time_PPG`
- `PPG_Ch1` (green)
- `PPG_Ch2` (infrared)
- `PPG_Ch3` (red)

Minor header variants such as `PPG1` and `PPG Ch1` are accepted. Rows with a
missing/non-numeric value in any required column are excluded.

## Outputs

- `*_ppg_analysis.xlsx`: heart rate, breathing rate, SpO2 estimate, ratio,
  breathing amplitude/count, and PPG-derived RMSSD/SDNN for each window.
- `*_ppg_filtered.xlsx`: filtered green heartbeat, green respiratory, infrared,
  and red waveforms.

These are all derived from PPG. The project does not analyze EEG, ECG, GSR,
temperature, or accelerometer data.

## Run

```bash
pip install -r requirements.txt
python ppg_analysis.py INPUT.xlsx --sampling-rate 95 --window-seconds 30
```

Use `--output-dir PATH` to choose the output folder and `--overlap 0.5` for 50%
window overlap. Run `python ppg_analysis.py --help` for all options.

## Important interpretation note

RMSSD and SDNN are pulse-rate-variability estimates from PPG peak intervals,
not ECG-derived HRV. SpO2 is also an algorithmic estimate and requires
calibration/validation before physiological or clinical interpretation.
