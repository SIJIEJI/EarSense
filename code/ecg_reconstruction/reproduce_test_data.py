#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parents[1]
TEST_SCRIPT = ROOT / "ecg_reconstruct_test.py"
DEFAULT_CSV = RELEASE_ROOT / "data" / "ecg_reconstruction" / "test_data.csv"
DEFAULT_CKPT = RELEASE_ROOT / "models" / "ecg_reconstruction" / "best_model.pth"
OUT_DIR = RELEASE_ROOT / "results" / "ecg_reconstruction" / "reproduction_outputs"


def load_test_module():
    spec = importlib.util.spec_from_file_location("ecg_reconstruct_test", TEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def pearson_r(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    return cosine_similarity(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else RELEASE_ROOT / args.csv
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else RELEASE_ROOT / args.ckpt
    out_dir = args.out_dir if args.out_dir.is_absolute() else RELEASE_ROOT / args.out_dir
    prefix = args.prefix or csv_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path, header=None, names=["time", "ground_truth_ecg", "ear_input"])
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    time = data["time"].to_numpy(dtype=float)
    ground_truth = data["ground_truth_ecg"].to_numpy(dtype=float)
    ear_input = data["ear_input"].to_numpy(dtype=float)

    ecg_test = load_test_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ecg_test.ECGNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    reconstructed = ecg_test.reconstruct_full_ecg(
        mixed=ear_input,
        model=model,
        device=device,
        segment_length=2000,
        overlap=0.5,
        batch_size=64,
    )

    n = min(len(time), len(ear_input), len(ground_truth), len(reconstructed))
    time = time[:n]
    ear_input = ear_input[:n]
    ground_truth = ground_truth[:n]
    reconstructed = reconstructed[:n]

    mse = float(np.mean((ground_truth - reconstructed) ** 2))
    mae = float(np.mean(np.abs(ground_truth - reconstructed)))
    cos = cosine_similarity(ground_truth, reconstructed)
    r = pearson_r(ground_truth, reconstructed)

    out_csv = out_dir / f"{prefix}_reconstruction.csv"
    pd.DataFrame(
        {
            "time": time,
            "ear_input": ear_input,
            "reconstructed_ecg": reconstructed,
            "ground_truth_ecg": ground_truth,
        }
    ).to_csv(out_csv, index=False)

    plt.figure(figsize=(14, 4))
    plt.plot(time, ear_input, color="#4C4C4C", linewidth=1.0)
    plt.title("Ear Input Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    ear_plot = out_dir / f"{prefix}_figure_1_ear_data.png"
    plt.savefig(ear_plot, dpi=220)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(time, ground_truth, label="Ground truth ECG", color="#1f77b4", linewidth=1.0)
    plt.plot(time, reconstructed, label="Reconstructed ECG", color="#ff7f0e", linewidth=1.0, alpha=0.9)
    plt.title(f"ECG Reconstruction vs Ground Truth | CosSim={cos:.3f}, r={r:.3f}, MAE={mae:.4f}, MSE={mse:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    recon_plot = out_dir / f"{prefix}_figure_2_reconstruction_vs_ground_truth.png"
    plt.savefig(recon_plot, dpi=220)
    plt.close()

    print(f"device={device}")
    print(f"csv={csv_path}")
    print(f"samples={n}")
    print(f"mse={mse:.8f}")
    print(f"mae={mae:.8f}")
    print(f"cosine_similarity={cos:.8f}")
    print(f"pearson_r={r:.8f}")
    print(f"saved_csv={out_csv}")
    print(f"saved_ear_plot={ear_plot}")
    print(f"saved_reconstruction_plot={recon_plot}")


if __name__ == "__main__":
    main()
