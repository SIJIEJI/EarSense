# ECG Reconstruction

This folder contains ECGNet code for reconstructing a reference ECG waveform
from an in-ear composite biopotential signal.

## Recommended Reproduction Command

Run this from the repository root:

```bash
python code/ecg_reconstruction/reproduce_test_data.py
```

The script loads:

```text
data/ecg_reconstruction/test_data.csv
models/ecg_reconstruction/best_model.pth
```

and writes:

```text
results/ecg_reconstruction/reproduction_outputs/test_data_reconstruction.csv
results/ecg_reconstruction/reproduction_outputs/test_data_figure_1_ear_data.png
results/ecg_reconstruction/reproduction_outputs/test_data_figure_2_reconstruction_vs_ground_truth.png
```

Expected metrics for the included example are approximately:

```text
MSE = 0.0021
MAE = 0.0308
Cosine similarity = 0.9273
Pearson r = 0.9273
```

Small numerical differences can occur across CPU, CUDA, and PyTorch versions.

## Input CSV Format

`test_data.csv` is a headerless CSV with three columns:

```text
column 1: time
column 2: ground-truth/reference ECG
column 3: in-ear input signal
```

The generated reconstruction CSV contains:

```text
time
ear_input
reconstructed_ecg
ground_truth_ecg
```

## Scripts

Recommended:

```text
reproduce_test_data.py          End-to-end example using the released data and checkpoint
ecg_reconstruct_test.py         ECGNet definition and long-signal reconstruction utility
```

Training and extended analysis:

```text
ecg_training_improved.py        Training script for users with paired training data
ecg_recon_metrics_plus.py       Extended metrics and R-peak timing analysis
ecg_recon_rpeak_fix.py          R-peak evaluation helper
ecg_recon_with_errorbars.py     Metric plotting and confidence-interval helper
```

## Training Note

The public release includes one example test file and a pretrained checkpoint.
The full raw ECG training recordings are not included. To train with your own
paired in-ear/reference ECG recordings:

```bash
python ecg_training_improved.py --train_csvs train_subject1.csv train_subject2.csv --out_ckpt runs/ecg_best_model.pth
```

## Interpretation

R-peak timing helpers are useful for diagnostic plots, but the primary
reconstruction example is `reproduce_test_data.py`. Peak metrics depend on peak
detector settings and should be reported together with waveform-level metrics.
