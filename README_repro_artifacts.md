# EarSense Reproducibility Artifacts

This folder collects the code, trained model files, processed data, and result
tables/figures that were found under `D:\VUA\WJ` and matched the manuscript:

`Integrated brain-body physiological monitoring from a smart earbud`

The goal is to provide a clean staging folder before preparing a public GitHub
release. It is not yet a polished repository.

## What Is Included

### Sleep staging, Fig. 5G/H and fig. S58

- Code:
  - `code/sleep_stage/preprocess-ctx.py`
  - `code/sleep_stage/N1-ctx.py`
  - `code/sleep_stage/N1-ctx-overfit.py`
- Processed data:
  - `data/sleep_stage_processed/X.npy`
  - `data/sleep_stage_processed/y.npy`
  - `data/sleep_stage_processed/sid.npy`
  - `data/sleep_stage_processed/meta.json`
- Models:
  - `models/sleep_stage/model.pt`
  - `models/sleep_stage/model_all_overfit.pt`
  - `models/sleep_stage/stage1_n1_vs_rest.pt`
  - `models/sleep_stage/stage2_rest4.pt`
- Results:
  - `results/sleep_stage/SleepStageOutput/`
  - `results/sleep_stage/cascade_ctx_out/`

The manuscript reports sleep staging agreement near 89.9%. The matching copied
result is in `results/sleep_stage/SleepStageOutput/metrics.json`.

Example run from this folder:

```bash
python code/sleep_stage/N1-ctx.py --data_dir data/sleep_stage_processed --out_dir runs/sleep_stage_ctx --ctx 5 --cascade_mode soft --alpha_n1 1.45
```

### Sleep-derived questionnaire outcomes, Fig. 5I-K and fig. S59

- Code:
  - `code/questionnaire_outcomes/preprocess-Q.py`
  - `code/questionnaire_outcomes/aggreQ_5mins.py`
  - `code/questionnaire_outcomes/Q_train_classification_v2.py`
  - `code/questionnaire_outcomes/Q_train_multi.py`
  - `code/questionnaire_outcomes/LOSO_test.py`
  - `code/questionnaire_outcomes/train_full.py`
  - `code/questionnaire_outcomes/fewer_featrues.py`
  - `code/utilities/plot_mlt.py`
- Processed data:
  - `data/questionnaire_processed/nn_dataset_subject_level.npz`
  - `data/questionnaire_processed/nn_dataset_subject_level_5min.npz`
  - `data/questionnaire_processed/sleep_questionnaires_scores_one_row_per_subject_class.xlsx`
  - `data/questionnaire_processed/sleep_questionnaires_scores_one_row_per_subject.xlsx`
  - `data/questionnaire_processed/label.csv`
  - `data/questionnaire_processed/modalities.json`
  - `data/questionnaire_processed/modalities-EEGonly.json`
  - `data/questionnaire_processed/build_meta.json`
- Results:
  - `results/questionnaire_outcomes/ess_loso_eval/`
  - `results/questionnaire_outcomes/PSQI_loso_eval/`
  - `results/questionnaire_outcomes/FOSQ_loso_eval/`
  - `results/questionnaire_outcomes/ml_summary_out/`

Example run:

```bash
python code/questionnaire_outcomes/fewer_featrues.py --npz data/questionnaire_processed/nn_dataset_subject_level.npz --label_xlsx data/questionnaire_processed/sleep_questionnaires_scores_one_row_per_subject_class.xlsx --task_col FOSQ_class --mod_json data/questionnaire_processed/modalities-EEGonly.json --out_dir runs/FOSQ_eval --model logreg --normalize_cm
```

### PVT slowest 10 percent reaction-time prediction, Fig. 5L and fig. S60

- Code:
  - `code/pvt/pvt_dl_v4.py`
  - `code/pvt/pvtmean.py`
  - `code/pvt/PVTregression.py`
  - `code/pvt/PVTregressionenhanced.py`
  - `code/pvt/PVTregression95.py`
- Data:
  - Uses `data/questionnaire_processed/nn_dataset_subject_level.npz`
  - Uses `data/questionnaire_processed/sleep_questionnaires_scores_one_row_per_subject_class.xlsx`
- Model:
  - `models/pvt/simple_mlp_weights.npz`
- Results:
  - `results/pvt/pvt_mlp_all_train_out/`
  - `results/pvt/loso_lapse_hgb_cqr/`

Example run:

```bash
python code/pvt/pvt_dl_v4.py --npz data/questionnaire_processed/nn_dataset_subject_level.npz --label_xlsx data/questionnaire_processed/sleep_questionnaires_scores_one_row_per_subject_class.xlsx --target_col mean_slowest_10pct_ms --out_dir runs/pvt_mlp_all_train
```

### ECG reconstruction, Fig. 2L-N and figs. S38-S42

- Code:
  - `code/ecg_reconstruction/ecg_training_improved.py`
  - `code/ecg_reconstruction/ecg_training_improved_v2.py`
  - `code/ecg_reconstruction/ecg_reconstruct_test.py`
  - `code/ecg_reconstruction/ecg_reconstruct_test_no_numpy.py`
  - `code/ecg_reconstruction/ecg_recon_metrics_plus.py`
  - `code/ecg_reconstruction/ecg_recon_with_errorbars.py`
  - `code/ecg_reconstruction/ecg_recon_rpeak_fix.py`
- Data:
  - `data/ecg_reconstruction/1.csv`
  - `data/ecg_reconstruction/2.csv`
  - `data/ecg_reconstruction/3.csv`
- Model:
  - `models/ecg_reconstruction/best_model.pth`
- Results:
  - `results/ecg_reconstruction/`

Example run:

```bash
python code/ecg_reconstruction/ecg_training_improved.py
```

Note: this ECG script expects `1.csv` and `2.csv` in the current working
directory. Either run it from `data/ecg_reconstruction/` after copying the
script there, or update the file paths before publishing.

## What Was Not Copied

- Raw sleep EDF folders under `D:\VUA\WJ\Sleep patient with sleep stage\` were
  not copied because folder names and raw files appear to contain participant
  identifiers. The processed arrays needed for direct reproduction were copied.
- The very large archive `D:\VUA\WJ\Successful test - Copy.zip` was not copied.
- Affective-state artifacts for Fig. 4 were not found in this folder by search
  terms including `PANAS`, `STAI`, `affective`, `LGBM`, `LightGBM`, `SHAP`,
  `CPT`, and `VR`. Add those scripts/data separately if they are stored outside
  `D:\VUA\WJ`.

## Suggested Dependencies

Use `requirements_repro.txt` for a fuller dependency list than the original
`requirements_original.txt`.

## Manuscript Reference

- `docs/Combined Manuscript_20260323.docx`
- `docs/manuscript_text_extracted.txt`

