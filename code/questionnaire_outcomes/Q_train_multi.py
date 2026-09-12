#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
)


# ----------------------------
# Dataset
# ----------------------------
class SubjectSeqMTLDataset(Dataset):
    def __init__(self, X, mask, y_reg, y_cls):
        self.X = torch.from_numpy(X).float()
        self.m = torch.from_numpy(mask).float()
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_cls = torch.from_numpy(y_cls).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.m[idx], self.y_reg[idx], self.y_cls[idx]


class SubjectSeqMTLDatasetWithID(Dataset):
    def __init__(self, X, mask, y_reg, y_cls, sid):
        self.X = torch.from_numpy(X).float()
        self.m = torch.from_numpy(mask).float()
        self.y_reg = torch.from_numpy(y_reg).float()
        self.y_cls = torch.from_numpy(y_cls).long()
        self.sid = torch.from_numpy(sid).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.m[idx], self.y_reg[idx], self.y_cls[idx], self.sid[idx]


# ----------------------------
# Model: shared encoder + 2 heads
# ----------------------------
class TemporalCNN_BiGRU_MTL(nn.Module):
    def __init__(self, in_dim, n_classes, cnn_channels=128, gru_hidden=128, gru_layers=1, dropout=0.25,
                 cls_use_regpred=False):
        super().__init__()
        self.cls_use_regpred = bool(cls_use_regpred)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        feat_dim = gru_hidden * 2

        # regression head
        self.reg_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1)
        )

        # classification head
        cls_in = feat_dim + (1 if self.cls_use_regpred else 0)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_in),
            nn.Linear(cls_in, gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, n_classes)
        )

    def encode(self, x, mask):
        # x: (B,T,F), mask: (B,T)
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B,T,C)
        h, _ = self.gru(z)                               # (B,T,2H)
        m = mask.unsqueeze(-1)                           # (B,T,1)
        h = h * m
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom                    # (B,2H)
        return pooled

    def forward(self, x, mask):
        feat = self.encode(x, mask)                      # (B,2H)
        reg = self.reg_head(feat).squeeze(-1)            # (B,)

        if self.cls_use_regpred:
            feat2 = torch.cat([feat, reg.unsqueeze(-1)], dim=-1)
        else:
            feat2 = feat
        logits = self.cls_head(feat2)                    # (B,C)

        return reg, logits


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_classes(y_raw: np.ndarray):
    """Map arbitrary labels -> 0..C-1"""
    uniq = pd.unique(y_raw)
    try:
        uniq_sorted = sorted(uniq, key=lambda x: float(x))
    except Exception:
        uniq_sorted = sorted(uniq, key=lambda x: str(x))
    mapping = {v: i for i, v in enumerate(uniq_sorted)}
    inv = {i: v for v, i in mapping.items()}
    y_enc = np.array([mapping[v] for v in y_raw], dtype=np.int64)
    return y_enc, mapping, inv


def compute_class_weights(y_cls: np.ndarray, n_classes: int):
    counts = np.bincount(y_cls, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def fit_standardizer(X_tr: np.ndarray, mask_tr: np.ndarray):
    m = mask_tr.astype(bool)
    count = m.sum()
    if count == 0:
        mu = X_tr.mean(axis=(0, 1))
        sd = X_tr.std(axis=(0, 1)) + 1e-6
        return mu.astype(np.float32), sd.astype(np.float32)
    mb = m[..., None]
    s1 = (X_tr * mb).sum(axis=(0, 1))
    s2 = ((X_tr ** 2) * mb).sum(axis=(0, 1))
    mu = s1 / float(count)
    var = s2 / float(count) - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-6))
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return ((X - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)


# ----------------------------
# Train / Eval (no early stop)
# ----------------------------
def train_one_fold_mtl(model, dl_tr, dl_va, device,
                       lr=1e-3, wd=1e-4, epochs=400, log_every=100,
                       lambda_reg=1.0, lambda_cls=1.0,
                       class_weights=None, label_smoothing=0.0,
                       reg_zscore=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    reg_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=float(label_smoothing)
    )

    # (optional) standardize regression target inside the fold to stabilize training
    y_mu, y_sd = 0.0, 1.0
    if reg_zscore:
        ys = []
        for _, _, yb_reg, _ in dl_tr:
            ys.append(yb_reg.numpy())
        ys = np.concatenate(ys)
        y_mu = float(np.mean(ys))
        y_sd = float(np.std(ys) + 1e-6)

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, mb, yb_reg, yb_cls in dl_tr:
            xb, mb = xb.to(device), mb.to(device)
            yb_reg = yb_reg.to(device)
            yb_cls = yb_cls.to(device)

            pred_reg, logits = model(xb, mb)

            if reg_zscore:
                yb_reg_n = (yb_reg - y_mu) / y_sd
                pred_reg_n = (pred_reg - y_mu) / y_sd
                loss_reg = reg_loss_fn(pred_reg_n, yb_reg_n)
            else:
                loss_reg = reg_loss_fn(pred_reg, yb_reg)

            loss_cls = cls_loss_fn(logits, yb_cls)
            loss = lambda_reg * loss_reg + lambda_cls * loss_cls

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        if (log_every is not None) and (ep % log_every == 0 or ep == 1 or ep == epochs):
            tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
            if dl_va is not None:
                model.eval()
                with torch.no_grad():
                    va_losses = []
                    for xb, mb, yb_reg, yb_cls in dl_va:
                        xb, mb = xb.to(device), mb.to(device)
                        yb_reg = yb_reg.to(device)
                        yb_cls = yb_cls.to(device)
                        pr, lg = model(xb, mb)
                        if reg_zscore:
                            yb_reg_n = (yb_reg - y_mu) / y_sd
                            pr_n = (pr - y_mu) / y_sd
                            lreg = reg_loss_fn(pr_n, yb_reg_n)
                        else:
                            lreg = reg_loss_fn(pr, yb_reg)
                        lcls = cls_loss_fn(lg, yb_cls)
                        va_losses.append((lambda_reg * lreg + lambda_cls * lcls).item())
                    va = float(np.mean(va_losses)) if va_losses else float("nan")
                print(f"  Epoch {ep:4d}/{epochs}  train_loss={tr:.4f}  val_loss={va:.4f}")
            else:
                print(f"  Epoch {ep:4d}/{epochs}  train_loss={tr:.4f}")

    return model


@torch.no_grad()
def predict_mtl(model, dl, device):
    model.eval()
    pred_reg, true_reg, pred_cls, true_cls, ids = [], [], [], [], []
    for xb, mb, yb_reg, yb_cls, sid in dl:
        xb, mb = xb.to(device), mb.to(device)
        pr, lg = model(xb, mb)
        pc = torch.argmax(lg, dim=1).cpu().numpy()

        pred_reg.append(pr.cpu().numpy())
        true_reg.append(yb_reg.numpy())
        pred_cls.append(pc)
        true_cls.append(yb_cls.numpy())
        ids.append(sid.numpy())

    return (np.concatenate(pred_reg), np.concatenate(true_reg),
            np.concatenate(pred_cls), np.concatenate(true_cls),
            np.concatenate(ids))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level_5min.npz")
    ap.add_argument("--out_dir", default="./cv_mtl_psqi_out")

    # regression target from npz label_names
    ap.add_argument("--reg_target", default="PSQI_raw_sum", help="regression target label name in npz label_names")

    # classification labels from xlsx
    ap.add_argument("--cls_xlsx", required=False, default="sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col", default="subject", help="subject id column in xlsx")
    ap.add_argument("--cls_col", required=False, default="PSQI_class", help="classification label column name in xlsx")

    ap.add_argument("--cv", choices=["kfold", "stratified"], default="stratified")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=200)

    ap.add_argument("--cnn_channels", type=int, default=128)
    ap.add_argument("--gru_hidden", type=int, default=128)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.25)

    # loss weights
    ap.add_argument("--lambda_reg", type=float, default=1.0)
    ap.add_argument("--lambda_cls", type=float, default=1.0)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--reg_zscore", action="store_true")
    ap.add_argument("--cls_use_regpred", action="store_true", help="append reg_pred into cls head input")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # load npz
    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    mask = data["mask"].astype(np.float32)
    y_all = data["y"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(str)
    label_names = data["label_names"].astype(str)

    if args.reg_target not in list(label_names):
        raise ValueError(f"--reg_target {args.reg_target} not in label_names: {list(label_names)}")
    reg_idx = list(label_names).index(args.reg_target)
    y_reg_full = y_all[:, reg_idx].astype(np.float32)

    # load xlsx classification labels
    df = pd.read_excel(args.cls_xlsx)
    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found in xlsx columns: {list(df.columns)}")
    if args.cls_col not in df.columns:
        raise ValueError(f"cls_col '{args.cls_col}' not found in xlsx columns: {list(df.columns)}")

    df[args.id_col] = df[args.id_col].astype(str)
    df = df.dropna(subset=[args.cls_col])
    label_map = df.set_index(args.id_col)

    # align subjects
    keep = []
    y_cls_raw = []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep.append(i)
            y_cls_raw.append(label_map.loc[sid, args.cls_col])
    keep = np.array(keep, dtype=np.int64)

    if len(keep) == 0:
        raise ValueError("No overlapping subjects between npz subject_ids and xlsx subject column.")

    X = X[keep]
    mask = mask[keep]
    subject_ids = subject_ids[keep]
    y_reg = y_reg_full[keep]

    y_cls_raw = np.array(y_cls_raw)
    y_cls, mapping, inv = encode_classes(y_cls_raw)
    n_classes = int(y_cls.max() + 1)

    # choose splitter
    if args.cv == "stratified":
        counts = np.bincount(y_cls, minlength=n_classes)
        min_count = int(counts.min())
        k_use = min(args.k, min_count)
        if k_use < 2:
            raise ValueError(f"Some class has <2 samples; cannot CV. class_counts={counts.tolist()}")
        splitter = StratifiedKFold(n_splits=k_use, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.arange(len(y_cls)), y_cls)
        k_final = k_use
    else:
        splitter = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.arange(len(y_cls)))
        k_final = args.k

    N, T, F = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # store OOF predictions
    oof_reg = np.zeros((N,), dtype=np.float32)
    oof_cls = np.zeros((N,), dtype=np.int64)

    fold_reports = []

    for fold, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        tr_idx = np.array(tr_idx)
        te_idx = np.array(te_idx)

        # simple val split from train (monitor only)
        # shuffle tr_idx for fair val
        rng = np.random.RandomState(args.seed + fold)
        tr_perm = tr_idx.copy()
        rng.shuffle(tr_perm)
        n_val = max(2, int(0.1 * len(tr_perm)))
        val_idx = tr_perm[:n_val]
        train_idx = tr_perm[n_val:]

        # fold-wise feature standardization (important!)
        mu, sd = fit_standardizer(X[train_idx], mask[train_idx])
        X_tr = apply_standardizer(X[train_idx], mu, sd)
        X_va = apply_standardizer(X[val_idx], mu, sd)
        X_te = apply_standardizer(X[te_idx], mu, sd)

        ds_tr = SubjectSeqMTLDataset(X_tr, mask[train_idx], y_reg[train_idx], y_cls[train_idx])
        ds_va = SubjectSeqMTLDataset(X_va, mask[val_idx], y_reg[val_idx], y_cls[val_idx])
        ds_te = SubjectSeqMTLDatasetWithID(X_te, mask[te_idx], y_reg[te_idx], y_cls[te_idx], te_idx.astype(np.int64))

        bs = min(args.batch_size, len(ds_tr)) if len(ds_tr) > 0 else 1
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=1, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=1, shuffle=False)

        print(f"\n[Fold {fold}/{k_final}] train={len(train_idx)} val={len(val_idx)} test={len(te_idx)}")

        class_w = compute_class_weights(y_cls[train_idx], n_classes) if args.use_class_weights else None

        model = TemporalCNN_BiGRU_MTL(
            in_dim=F, n_classes=n_classes,
            cnn_channels=args.cnn_channels,
            gru_hidden=args.gru_hidden,
            gru_layers=args.gru_layers,
            dropout=args.dropout,
            cls_use_regpred=args.cls_use_regpred
        ).to(device)

        model = train_one_fold_mtl(
            model, dl_tr, dl_va, device,
            lr=args.lr, wd=args.wd,
            epochs=args.epochs, log_every=args.log_every,
            lambda_reg=args.lambda_reg, lambda_cls=args.lambda_cls,
            class_weights=class_w,
            label_smoothing=args.label_smoothing,
            reg_zscore=args.reg_zscore
        )

        pr, tr, pc, tc, idxs = predict_mtl(model, dl_te, device)
        oof_reg[idxs] = pr.astype(np.float32)
        oof_cls[idxs] = pc.astype(np.int64)

        # regression metrics
        mae = mean_absolute_error(tr, pr)
        rmse = float(np.sqrt(mean_squared_error(tr, pr)))
        r2 = float(r2_score(tr, pr)) if len(tr) >= 3 else float("nan")

        # classification metrics
        acc = float(accuracy_score(tc, pc))
        bal = float(balanced_accuracy_score(tc, pc))
        mf1 = float(f1_score(tc, pc, average="macro"))
        cm = confusion_matrix(tc, pc, labels=list(range(n_classes)))

        fold_reports.append({
            "fold": fold,
            "reg_MAE": float(mae),
            "reg_RMSE": float(rmse),
            "reg_R2": float(r2),
            "cls_acc": acc,
            "cls_bal_acc": bal,
            "cls_macro_f1": mf1,
            "cls_confusion_matrix": cm.tolist(),
            "n_test": int(len(tc))
        })

        print(f"[Fold {fold}] REG: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} | "
              f"CLS: acc={acc:.3f} bal_acc={bal:.3f} macro_f1={mf1:.3f}")

    # overall OOF metrics
    overall = {
        "N": int(N),
        "reg_target": args.reg_target,
        "cls_col": args.cls_col,
        "cv": args.cv,
        "k": int(k_final),
        "loss_weights": {"lambda_reg": args.lambda_reg, "lambda_cls": args.lambda_cls},
        "cls_mapping": {str(k): int(v) for k, v in mapping.items()},
        "reg_overall": {
            "MAE": float(mean_absolute_error(y_reg, oof_reg)),
            "RMSE": float(np.sqrt(mean_squared_error(y_reg, oof_reg))),
            "R2": float(r2_score(y_reg, oof_reg)),
        },
        "cls_overall": {
            "acc": float(accuracy_score(y_cls, oof_cls)),
            "bal_acc": float(balanced_accuracy_score(y_cls, oof_cls)),
            "macro_f1": float(f1_score(y_cls, oof_cls, average="macro")),
            "confusion_matrix": confusion_matrix(y_cls, oof_cls, labels=list(range(n_classes))).tolist()
        },
        "note": "Shared encoder + regression head + classification head (multi-task)."
    }

    with open(os.path.join(args.out_dir, "fold_metrics.json"), "w") as f:
        json.dump(fold_reports, f, indent=2)
    with open(os.path.join(args.out_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall, f, indent=2)

    # save predictions
    out_csv = os.path.join(args.out_dir, "predictions_oof.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("subject_id,true_reg,pred_reg,true_cls,pred_cls\n")
        for i in range(N):
            f.write(f"{subject_ids[i]},{y_reg[i]},{oof_reg[i]},{int(y_cls[i])},{int(oof_cls[i])}\n")

    print("\n[Overall]")
    print(json.dumps(overall, indent=2))
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
