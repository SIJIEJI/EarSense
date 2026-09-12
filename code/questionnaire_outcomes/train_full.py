#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Label encoding (ordinal)
# -------------------------
def encode_classes_ordinal(y_raw):
    uniq = pd.unique(y_raw)
    try:
        uniq_sorted = sorted(uniq, key=lambda x: float(x))
    except Exception:
        uniq_sorted = sorted(uniq, key=lambda x: str(x))
    mapping = {v: i for i, v in enumerate(uniq_sorted)}
    inv = {i: v for v, i in mapping.items()}
    y = np.array([mapping[v] for v in y_raw], dtype=np.int64)
    return y, mapping, inv


# -------------------------
# Confusion matrix plot
# -------------------------
def plot_confusion(cm, labels, title, out_path, normalize=True):
    cm = np.array(cm, dtype=np.float32)
    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        cm = cm / rs

    plt.figure(figsize=(6.8, 5.8))
    plt.imshow(cm, vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


# -------------------------
# Normalization stats
# -------------------------
def masked_mean_std_over_all(X, mask):
    m = mask.astype(bool)[..., None]  # (N,T,1)
    cnt = m.sum(axis=(0, 1)).clip(min=1)
    mu = (X * m).sum(axis=(0, 1)) / cnt
    mu2 = ((X ** 2) * m).sum(axis=(0, 1)) / cnt
    var = np.maximum(mu2 - mu ** 2, 1e-6)
    sd = np.sqrt(var)
    return mu.astype(np.float32), sd.astype(np.float32)


# -------------------------
# Dataset
# -------------------------
class SubjectSeqDataset(Dataset):
    def __init__(self, X, mask, y, mean=None, std=None):
        self.X = X.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        m = self.mask[idx]
        if (self.mean is not None) and (self.std is not None):
            x = (x - self.mean[None, :]) / (self.std[None, :] + 1e-6)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(m).float(),
            torch.tensor(self.y[idx]).long()
        )


# -------------------------
# CORAL helpers
# -------------------------
def coral_targets(y, num_classes):
    B = y.shape[0]
    K = num_classes
    t = torch.zeros((B, K - 1), device=y.device, dtype=torch.float32)
    for j in range(K - 1):
        t[:, j] = (y >= (j + 1)).float()
    return t


def coral_loss(logits, y, num_classes, pos_weight=None):
    t = coral_targets(y, num_classes=num_classes)
    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(logits, t)


@torch.no_grad()
def coral_predict_class(logits):
    p = torch.sigmoid(logits)
    passed = (p > 0.5).sum(dim=1)
    return passed.long()


def compute_threshold_pos_weight(y_train, num_classes, device):
    y = torch.tensor(y_train, dtype=torch.long)
    K = num_classes
    pw = []
    for j in range(K - 1):
        pos = (y >= (j + 1)).sum().item()
        neg = len(y_train) - pos
        pos = max(pos, 1)
        neg = max(neg, 1)
        pw.append(neg / pos)
    return torch.tensor(pw, dtype=torch.float32, device=device)


# -------------------------
# Model
# -------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, c, k=5, dropout=0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=pad)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad)
        self.bn2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return F.gelu(h + x)


class DeepOrdinalNet(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes=5,
        cnn_channels=384,
        n_res_blocks=6,
        gru_hidden=384,
        gru_layers=2,
        dropout=0.0,
        use_reg_head=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_reg_head = use_reg_head

        self.stem1 = nn.Sequential(
            nn.Conv1d(in_dim, cnn_channels, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.stem2 = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        self.res = nn.Sequential(*[
            ResidualConvBlock(cnn_channels, k=5, dropout=dropout)
            for _ in range(n_res_blocks)
        ])

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.attn = nn.Linear(gru_hidden * 2, 1)

        self.ord_head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, num_classes - 1)
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1)
        )

    def _downsample_mask(self, mask, stride):
        m = mask.unsqueeze(1)  # (B,1,T)
        m = F.max_pool1d(m, kernel_size=stride, stride=stride, ceil_mode=True)
        return m.squeeze(1)

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # (B,F,T)

        x = self.stem1(x)
        mask = self._downsample_mask(mask, stride=2)

        x = self.stem2(x)
        mask = self._downsample_mask(mask, stride=2)

        x = self.res(x)        # (B,C,T')
        x = x.transpose(1, 2)  # (B,T',C)

        h, _ = self.gru(x)     # (B,T',2H)

        scores = self.attn(h).squeeze(-1)  # (B,T')
        scores = scores.masked_fill(mask <= 0.0, -1e9)
        w = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)        # (B,2H)

        logits_ord = self.ord_head(pooled) # (B,K-1)
        pred_reg = self.reg_head(pooled).squeeze(-1) if self.use_reg_head else None
        return logits_ord, pred_reg


# -------------------------
# Sampler
# -------------------------
def make_weighted_sampler(y, num_classes, power=1.0):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    inv = (counts.sum() / np.maximum(counts, 1.0)) ** power
    w = inv[y]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(w, dtype=torch.double),
        num_samples=len(y),
        replacement=True
    )
    return sampler, counts, inv


# -------------------------
# Train / Eval
# -------------------------
def train_full(model, dl, device, num_classes, epochs, lr, wd, log_every, lambda_reg, pos_weight):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        preds, trues = [], []

        for xb, mb, yb in dl:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            logits_ord, pred_reg = model(xb, mb)

            loss_ord = coral_loss(logits_ord, yb, num_classes=num_classes, pos_weight=pos_weight)

            if pred_reg is not None:
                y_reg = yb.float() + 1.0
                loss_reg = F.smooth_l1_loss(pred_reg, y_reg)
                loss = loss_ord + lambda_reg * loss_reg
            else:
                loss = loss_ord

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            preds.append(coral_predict_class(logits_ord).detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = accuracy_score(trues, preds)

        if (ep == 1) or (ep % log_every == 0) or (ep == epochs):
            print(f"Epoch {ep:4d}/{epochs}  train_loss={np.mean(losses):.4f}  train_acc={acc:.3f}")

    return model


@torch.no_grad()
def eval_full(model, dl, device, num_classes):
    model.eval()
    preds, trues = [], []
    for xb, mb, yb in dl:
        xb, mb = xb.to(device), mb.to(device)
        logits_ord, _ = model(xb, mb)
        preds.append(coral_predict_class(logits_ord).cpu().numpy())
        trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = float(accuracy_score(trues, preds))
    bal = float(balanced_accuracy_score(trues, preds))
    mf1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
    return preds, trues, acc, bal, mf1, cm


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--task_col", default="PSQI_class")
    ap.add_argument("--out_dir", default="./PSQI_overfit_showcase")

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--cnn_channels", type=int, default=384)
    ap.add_argument("--n_res_blocks", type=int, default=6)
    ap.add_argument("--gru_hidden", type=int, default=384)
    ap.add_argument("--gru_layers", type=int, default=2)

    ap.add_argument("--use_reg_head", action="store_true")
    ap.add_argument("--lambda_reg", type=float, default=0.6)

    ap.add_argument("--sampler_power", type=float, default=1.8)
    ap.add_argument("--use_threshold_pos_weight", action="store_true")

    ap.add_argument("--normalize_cm", action="store_true")
    ap.add_argument("--log_every", type=int, default=200)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load npz
    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    mask = d["mask"].astype(np.float32)
    subject_ids = d["subject_ids"].astype(str)
    feature_names = d["feature_names"].astype(str).tolist() if "feature_names" in d.files else None

    # Load labels
    df = pd.read_excel(args.label_xlsx)
    df[args.id_col] = df[args.id_col].astype(str)
    if args.task_col not in df.columns:
        raise ValueError(f"{args.task_col} not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[args.task_col])
    label_map = df.set_index(args.id_col)

    keep = []
    y_raw = []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep.append(i)
            y_raw.append(label_map.loc[sid, args.task_col])

    keep = np.array(keep, dtype=np.int64)
    X = X[keep]
    mask = mask[keep]
    subject_ids = subject_ids[keep]
    y_raw = np.array(y_raw)

    y, mapping, inv = encode_classes_ordinal(y_raw)
    num_classes = int(y.max() + 1)

    counts = np.bincount(y, minlength=num_classes)
    print("N:", len(y))
    print("class_counts(idx):", {int(i): int(c) for i, c in enumerate(counts)})
    print("classes(idx->raw):", inv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T, Fdim = X.shape

    # Norm stats (full data, since we overfit)
    mu, sd = masked_mean_std_over_all(X, mask)

    ds = SubjectSeqDataset(X, mask, y, mean=mu, std=sd)
    sampler, _, invw = make_weighted_sampler(y, num_classes=num_classes, power=args.sampler_power)

    bs = min(args.batch_size, len(ds))
    dl_train = DataLoader(ds, batch_size=bs, sampler=sampler)
    dl_eval = DataLoader(ds, batch_size=1, shuffle=False)

    pos_weight = None
    if args.use_threshold_pos_weight:
        pos_weight = compute_threshold_pos_weight(y, num_classes=num_classes, device=device)

    model = DeepOrdinalNet(
        in_dim=Fdim,
        num_classes=num_classes,
        cnn_channels=args.cnn_channels,
        n_res_blocks=args.n_res_blocks,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        use_reg_head=args.use_reg_head
    ).to(device)

    # Train
    model = train_full(
        model, dl_train, device,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        log_every=args.log_every,
        lambda_reg=args.lambda_reg,
        pos_weight=pos_weight
    )

    # Eval on SAME data (overfit showcase)
    pred, true, acc, bal, mf1, cm = eval_full(model, dl_eval, device, num_classes=num_classes)

    labels_show = [str(inv[i]) for i in range(num_classes)]
    cm_path = os.path.join(args.out_dir, "confusion_matrix_train.png")
    plot_confusion(
        cm, labels_show,
        f"{args.task_col} TRAIN (overfit showcase) | acc={acc:.3f} bal={bal:.3f} macroF1={mf1:.3f}",
        cm_path,
        normalize=args.normalize_cm
    )

    # Save predictions
    out_csv = os.path.join(args.out_dir, "predictions_train.csv")
    rows = []
    for i in range(N):
        rows.append({
            "subject_id": subject_ids[i],
            "true_idx": int(true[i]),
            "pred_idx": int(pred[i]),
            "true_raw": inv[int(true[i])],
            "pred_raw": inv[int(pred[i])]
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Save model checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "mean": mu,
        "std": sd,
        "num_classes": num_classes,
        "class_mapping_raw_to_idx": {str(k): int(v) for k, v in mapping.items()},
        "class_mapping_idx_to_raw": {str(int(k)): inv[int(k)] for k in inv.keys()},
        "feature_names": feature_names,
        "args": vars(args)
    }
    ckpt_path = os.path.join(args.out_dir, "checkpoint_overfit_train.pt")
    torch.save(ckpt, ckpt_path)

    # Save metrics
    metrics = {"acc": acc, "bal_acc": bal, "macro_f1": mf1, "N": int(N)}
    with open(os.path.join(args.out_dir, "metrics_train.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    def to_py(x):
    # 把 numpy 标量/数组都变成 python 可序列化
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        return x

    with open(os.path.join(args.out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({
            "raw_to_idx": {str(to_py(k)): int(to_py(v)) for k, v in mapping.items()},
            "idx_to_raw": {str(int(to_py(k))): to_py(inv[int(to_py(k))]) for k in inv.keys()}
        }, f, indent=2, ensure_ascii=False)
    

    print("\n[TRAIN / OVERFIT SHOWCASE]")
    print(metrics)
    print("Saved:", cm_path)
    print("Saved:", out_csv)
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
