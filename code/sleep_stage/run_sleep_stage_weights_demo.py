#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run released sleep-stage model weights on a small processed demo subset.

Default behavior:
  - loads cascade weights from ../../models/sleep_stage/
  - reads demo data from ../../data/sleep_stage_demo_subset/
  - writes metrics, predictions, and confusion matrix to
    ../../results/sleep_stage_demo/

Examples:
  python run_sleep_stage_weights_demo.py
  python run_sleep_stage_weights_demo.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run this demo. Install the release requirements, "
        "for example: pip install -r requirements_repro.txt"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
REST4_TO_STAGE = np.array([0, 2, 3, 4], dtype=np.int64)


def release_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unexpected checkpoint format in {path}")


def get_state_dict(ckpt: dict) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "state_dict", "model_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt
    raise ValueError("Could not find model weights in checkpoint.")


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    return dev


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Conv1d(ch, mid, kernel_size=1)
        self.fc2 = nn.Conv1d(mid, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2, keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 7,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=k,
            stride=1,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout(dropout)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.short is not None:
            identity = self.short(identity)
        return F.relu(out + identity)


class CascadeSleepCNN(nn.Module):
    """Architecture used by N1-ctx.py stage1/stage2 checkpoints."""

    def __init__(self, in_ch: int, num_classes: int, base: int = 32, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResBlock1D(base, base, k=7, stride=1, dilation=1, dropout=dropout),
            ResBlock1D(base, base, k=7, stride=1, dilation=1, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(base, base * 2, k=7, stride=2, dilation=1, dropout=dropout),
            ResBlock1D(base * 2, base * 2, k=7, stride=1, dilation=2, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResBlock1D(base * 2, base * 4, k=7, stride=2, dilation=2, dropout=dropout),
            ResBlock1D(base * 4, base * 4, k=7, stride=1, dilation=4, dropout=dropout),
        )
        self.stage4 = nn.Sequential(
            ResBlock1D(base * 4, base * 4, k=5, stride=1, dilation=8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x)


class ContextDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sid: np.ndarray | None,
        indices: Iterable[int],
        ctx: int = 3,
    ):
        if ctx < 1 or ctx % 2 != 1:
            raise ValueError("ctx must be an odd positive integer.")
        self.X = X
        self.y = y.astype(np.int64)
        self.sid = sid.astype(np.int64) if sid is not None else None
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.ctx = int(ctx)
        self.half = self.ctx // 2

        self.group = None
        self.pos = None
        if self.sid is not None:
            self.group = {}
            for s in np.unique(self.sid):
                self.group[int(s)] = np.where(self.sid == s)[0]
            self.pos = np.empty((len(self.X),), dtype=np.int64)
            for arr in self.group.values():
                for p, gidx in enumerate(arr):
                    self.pos[gidx] = p

    def __len__(self) -> int:
        return int(len(self.indices))

    def _context_indices(self, gidx: int) -> list[int]:
        if self.ctx == 1:
            return [gidx]
        if self.sid is None:
            return [
                int(min(max(k, 0), len(self.X) - 1))
                for k in range(gidx - self.half, gidx + self.half + 1)
            ]
        assert self.group is not None and self.pos is not None
        arr = self.group[int(self.sid[gidx])]
        p = int(self.pos[gidx])
        return [
            int(arr[min(max(k, 0), len(arr) - 1)])
            for k in range(p - self.half, p + self.half + 1)
        ]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gidx = int(self.indices[i])
        ctx_idx = self._context_indices(gidx)
        x = np.concatenate([self.X[j] for j in ctx_idx], axis=1).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(int(self.y[gidx]), dtype=torch.long),
            torch.tensor(gidx, dtype=torch.long),
        )


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, labels, indices = [], [], []
    for xb, yb, ib in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(yb.numpy())
        indices.append(ib.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(indices)


def cascade_soft(p_n1: np.ndarray, p_rest4: np.ndarray, alpha_n1: float) -> np.ndarray:
    p_n1_adj = np.clip(p_n1, 1e-8, 1.0) ** float(alpha_n1)
    p_rest = np.clip(1.0 - p_n1_adj, 0.0, 1.0)
    p5 = np.zeros((len(p_n1), 5), dtype=np.float64)
    p5[:, 1] = p_n1_adj
    p5[:, REST4_TO_STAGE] = p_rest[:, None] * p_rest4
    denom = p5.sum(axis=1, keepdims=True)
    p5 = p5 / np.maximum(denom, 1e-12)
    return p5.argmax(axis=1).astype(np.int64)


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for yt, yp in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[int(yt), int(yp)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    bal_acc = float(recall[present].mean()) if present.any() else 0.0
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(int).tolist(),
    }


def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(5), STAGE_NAMES)
    ax.set_yticks(range(5), STAGE_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_predictions_csv(out_path: Path, indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "epoch_in_demo", "true_label", "true_stage", "pred_label", "pred_stage"])
        for row_i, (idx, yt, yp) in enumerate(zip(indices, y_true, y_pred)):
            writer.writerow([int(idx), row_i, int(yt), STAGE_NAMES[int(yt)], int(yp), STAGE_NAMES[int(yp)]])


def evaluate_and_save(
    out_dir: Path,
    mode: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix_fixed(y_true, y_pred, num_classes=5)
    metric_values = metrics_from_confusion(cm)
    metrics = {
        "mode": mode,
        "num_epochs": int(len(y_true)),
        "stage_names": STAGE_NAMES,
        "accuracy": metric_values["accuracy"],
        "balanced_accuracy": metric_values["balanced_accuracy"],
        "macro_f1": metric_values["macro_f1"],
        "per_class_precision": metric_values["per_class_precision"],
        "per_class_recall": metric_values["per_class_recall"],
        "per_class_f1": metric_values["per_class_f1"],
        "per_class_support": metric_values["per_class_support"],
        "confusion_matrix_labels_0_to_4": cm.astype(int).tolist(),
        "data_dir": str(args.data_dir),
        "weights_dir": str(args.weights_dir),
        "ctx": int(args.ctx),
        "alpha_n1": float(args.alpha_n1),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_predictions_csv(out_dir / "predictions.csv", indices, y_true, y_pred)
    plot_confusion(cm, out_dir / "confusion_matrix.png", f"Sleep-stage demo ({mode})")
    return metrics


def run_cascade(args: argparse.Namespace, X: np.ndarray, y: np.ndarray, sid: np.ndarray | None, device: torch.device):
    ds = ContextDataset(X, y, sid, indices=np.arange(len(y)), ctx=args.ctx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stage1_ckpt = load_torch_checkpoint(args.weights_dir / "stage1_n1_vs_rest.pt", device)
    stage2_ckpt = load_torch_checkpoint(args.weights_dir / "stage2_rest4.pt", device)

    m1 = CascadeSleepCNN(in_ch=X.shape[1], num_classes=2, dropout=args.dropout).to(device)
    m2 = CascadeSleepCNN(in_ch=X.shape[1], num_classes=4, dropout=args.dropout).to(device)
    m1.load_state_dict(get_state_dict(stage1_ckpt), strict=True)
    m2.load_state_dict(get_state_dict(stage2_ckpt), strict=True)

    p1, y_true, indices = predict_proba(m1, loader, device)
    p2, _, _ = predict_proba(m2, loader, device)
    y_pred = cascade_soft(p1[:, 1], p2, alpha_n1=args.alpha_n1)
    return y_true, y_pred, indices


def main() -> None:
    root = release_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=Path, default=root / "models" / "sleep_stage")
    parser.add_argument("--data_dir", type=Path, default=root / "data" / "sleep_stage_demo_subset")
    parser.add_argument("--out_dir", type=Path, default=root / "results" / "sleep_stage_demo")
    parser.add_argument("--ctx", type=int, default=3, help="Number of neighboring epochs including the central epoch; default: 3 (90 seconds).")
    parser.add_argument("--alpha_n1", type=float, default=1.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=0.15)
    args = parser.parse_args()

    device = choose_device(args.device)
    X = np.load(args.data_dir / "X.npy", mmap_mode="r")
    y = np.load(args.data_dir / "y.npy").astype(np.int64)
    sid_path = args.data_dir / "sid.npy"
    sid = np.load(sid_path).astype(np.int64) if sid_path.exists() else None

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    y_true, y_pred, indices = run_cascade(args, X, y, sid, device)

    metrics = evaluate_and_save(args.out_dir, "cascade", y_true, y_pred, indices, args)
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
