# EarSense Reproducibility Artifacts

This repository contains selected code, processed example data, model weights,
and lightweight reproduction scripts for the EarSense analyses.

The public release is intended for reproducible inference examples and method
reference. Full raw recordings are not included.

## Quick Start

Install dependencies:

```bash
pip install -r requirements_repro.txt
```

Run the ECG reconstruction example:

```bash
python code/ecg_reconstruction/reproduce_test_data.py
```

Run the sleep-stage released-weight demo:

```bash
python code/sleep_stage/run_sleep_stage_weights_demo.py
```

## Repository Layout

```text
code/
  ecg_reconstruction/          ECGNet inference, training, and metrics scripts
  sleep_stage/                 Sleep preprocessing, training, and weight demo
  questionnaire_outcomes/      Questionnaire outcome feature/classification scripts
  utilities/                   Shared plotting utility

data/
  ecg_reconstruction/          Example ECG test CSV
  sleep_stage_demo_subset/     Small processed sleep subset for inference demo
  sleep_stage_processed/       Sleep labels/metadata; full X.npy is excluded
  questionnaire_processed/     Processed subject-level feature datasets

models/
  ecg_reconstruction/          ECGNet pretrained checkpoint
  sleep_stage/                 Sleep-stage checkpoints

results/
  ecg_reconstruction/          Reproduction outputs generated from the ECG demo
  sleep_stage_demo/            Cascade sleep demo outputs
```

## ECG Reconstruction

Recommended script:

```bash
python code/ecg_reconstruction/reproduce_test_data.py
```

This uses:

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

Expected example metrics:

```text
MSE = 0.0021
MAE = 0.0308
Cosine similarity = 0.9273
Pearson r = 0.9273
```

The ECG training script is included for users with their own paired in-ear and
reference ECG data. The full raw ECG training recordings are not included.
Example:

```bash
python code/ecg_reconstruction/ecg_training_improved.py --train_csvs train_subject1.csv train_subject2.csv --out_ckpt runs/ecg_best_model.pth
```

## Sleep Staging

Recommended released-weight demo:

```bash
python code/sleep_stage/run_sleep_stage_weights_demo.py
```

This loads:

```text
models/sleep_stage/stage1_n1_vs_rest.pt
models/sleep_stage/stage2_rest4.pt
```

and runs inference on:

```text
data/sleep_stage_demo_subset/
```

The demo subset contains 160 contiguous 30-second epochs and is intended as a
small smoke test for loading weights and generating predictions. It is not a
subject-independent validation benchmark.

The default context is `--ctx 3`: the preceding, central, and following
30-second epochs are concatenated along time, and the central epoch is
classified. Context stays within each subject; boundary epochs are repeated.
This matches the training script's default context.

With the same released weights, demo data, and `--alpha_n1 1.4`, the context
comparison produced:

| Context | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | --- | --- |
| 3 epochs (default) | 0.893750 | 0.799839 | 0.809457 |
| 5 epochs | 0.875000 | 0.743630 | 0.750886 |

The default was selected from this demo comparison. These results do not
establish subject-independent superiority or identify the original training
context of the released weights. No weights were retrained.

The full processed sleep tensor `data/sleep_stage_processed/X.npy` is excluded
from the GitHub release because it is large. Raw EDF recordings are also not
included. If you have raw EDF files and sleep-stage CSVs, use:

```bash
python code/sleep_stage/preprocess-ctx.py --root_dir raw_sleep_edf --out_dir sleep_dataset_built
```

`code/sleep_stage/N1-ctx.py` trains the two-stage cascade model on processed
`X.npy`, `y.npy`, and `sid.npy`. Its built-in split is epoch-level 70/30, not a
subject-held-out validation split.

## Questionnaire Outcomes

For public reproduction, prefer the cross-validated classical ML script:

```bash
python code/questionnaire_outcomes/fewer_featrues.py \
  --npz data/questionnaire_processed/nn_dataset_subject_level.npz \
  --label_xlsx data/questionnaire_processed/sleep_questionnaires_scores_one_row_per_subject_class.xlsx \
  --task_col FOSQ_class \
  --mod_json data/questionnaire_processed/modalities-EEGonly.json \
  --out_dir results/questionnaire_outcomes/FOSQ_cv_example \
  --model logreg \
  --normalize_cm
```

## Included and Excluded Data

Included:

```text
data/ecg_reconstruction/test_data.csv
data/sleep_stage_demo_subset/
data/sleep_stage_processed/y.npy
data/sleep_stage_processed/sid.npy
data/sleep_stage_processed/meta.json
data/questionnaire_processed/
models/
```

Excluded:

```text
Raw sleep EDF folders
Raw ECG training recordings
data/sleep_stage_processed/X.npy
Participant-identifying raw folders
Large intermediate result archives
```

## Notes for Readers

- The demo scripts are designed to verify that the released weights and example
  data can be loaded end-to-end.
- `ARTIFACT_MANIFEST.csv` records the artifact inventory used when assembling
  this release.
