#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion(cm, out_path, title, normalize=True):
    cm = np.array(cm, dtype=np.float32)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks(range(5), STAGE_NAMES)
    plt.yticks(range(5), STAGE_NAMES)

    for i in range(5):
        for j in range(5):
            txt = f"{cm[i, j]*100:.1f}%" if normalize else str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# --------- data with context ----------
class ContextDataset(Dataset):
    def __init__(self, X, y, sid=None, indices=None, ctx=1, augment=False):
        assert ctx % 2 == 1
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.sid = sid.astype(np.int64) if sid is not None else None
        self.ctx = ctx
        self.half = ctx // 2
        self.augment = augment

        self.idx = np.arange(len(X), dtype=np.int64) if indices is None else np.array(indices, dtype=np.int64)

        # Build per-subject ordering (assumes X is time-ordered within each subject)
        self.group = None
        self.pos = None
        if self.sid is not None:
            self.group = {}
            for s in np.unique(self.sid):
                self.group[s] = np.where(self.sid == s)[0]
            self.pos = np.empty((len(self.X),), dtype=np.int64)
            for s, arr in self.group.items():
                for p, gidx in enumerate(arr):
                    self.pos[gidx] = p

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gidx = self.idx[i]

        if self.ctx == 1:
            x = self.X[gidx]
        else:
            if self.sid is None:
                # fallback (not recommended)
                neigh = []
                for k in range(gidx - self.half, gidx + self.half + 1):
                    kk = min(max(k, 0), len(self.X) - 1)
                    neigh.append(kk)
                x = np.concatenate([self.X[j] for j in neigh], axis=1)
            else:
                s = self.sid[gidx]
                arr = self.group[s]
                p = self.pos[gidx]
                neigh = []
                for k in range(p - self.half, p + self.half + 1):
                    kk = min(max(k, 0), len(arr) - 1)
                    neigh.append(arr[kk])
                x = np.concatenate([self.X[j] for j in neigh], axis=1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[gidx], dtype=torch.long)


# --------- model (simple but strong) ----------
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Conv1d(ch, mid, 1)
        self.fc2 = nn.Conv1d(mid, ch, 1)

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, stride=1, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch, r=8)
        self.drop = nn.Dropout(dropout)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.short is not None:
            identity = self.short(identity)
        return F.relu(out + identity)


class EnhancedSleepCNN(nn.Module):
    def __init__(self, in_ch, num_classes=5, base=32, dropout=0.15):
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
            ResBlock1D(base, base*2, k=7, stride=2, dilation=1, dropout=dropout),
            ResBlock1D(base*2, base*2, k=7, stride=1, dilation=2, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResBlock1D(base*2, base*4, k=7, stride=2, dilation=2, dropout=dropout),
            ResBlock1D(base*4, base*4, k=7, stride=1, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base*4, base*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base*2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


def make_sampler(y, num_classes=5):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w_class = 1.0 / np.sqrt(counts)  # invsqrt helps stability
    w = w_class[y]
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(y), replacement=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    p_all, t_all = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        p_all.append(pred)
        t_all.append(yb.numpy())
    p = np.concatenate(p_all)
    t = np.concatenate(t_all)
    cm = confusion_matrix(t, p, labels=[0,1,2,3,4])
    acc = float(accuracy_score(t, p))
    bal = float(balanced_accuracy_score(t, p))
    mf1 = float(f1_score(t, p, average="macro"))
    kap = float(cohen_kappa_score(t, p, labels=[0,1,2,3,4]))
    kap_q = float(cohen_kappa_score(t, p, labels=[0,1,2,3,4], weights="quadratic"))
    return p, t, cm, acc, bal, mf1, kap, kap_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./sleep_dataset_built/merged")
    ap.add_argument("--out_dir", default="./overfit_all_showcase")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--normalize_cm", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    X = np.load(os.path.join(args.data_dir, "X.npy")).astype(np.float32)  # (N,C,T)
    y = np.load(os.path.join(args.data_dir, "y.npy")).astype(np.int64)
    valid = (y >= 0) & (y <= 4)
    X, y = X[valid], y[valid]

    sid_path = os.path.join(args.data_dir, "sid.npy")
    sid = np.load(sid_path).astype(np.int64)[valid] if os.path.exists(sid_path) else None

    if args.ctx > 1 and sid is None:
        print("[WARN] sid.npy not found; ctx>1 may cross subjects (not recommended).")

    print("[INFO] Using ALL data for training+eval (overfit showcase)")
    print("[INFO] X:", X.shape, "y:", y.shape)
    print("[INFO] counts:", dict(Counter(y.tolist())))

    ds = ContextDataset(X, y, sid=sid, indices=None, ctx=args.ctx, augment=True)
    sampler = make_sampler(y, num_classes=5)
    dl_tr = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, drop_last=False)

    ds_ev = ContextDataset(X, y, sid=sid, indices=None, ctx=args.ctx, augment=False)
    dl_ev = DataLoader(ds_ev, batch_size=args.batch_size*2, shuffle=False, drop_last=False)

    model = EnhancedSleepCNN(in_ch=X.shape[1], num_classes=5, base=32, dropout=args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        if ep == 1 or ep % 20 == 0 or ep == args.epochs:
            _, _, _, acc, _, _, kap, _ = evaluate(model, dl_ev)
            print(f"[ep {ep:03d}] loss={np.mean(losses):.4f}  agreement(acc)={acc:.3f}  kappa={kap:.3f}")

    # final eval
    pred, true, cm, acc, bal, mf1, kap, kap_q = evaluate(model, dl_ev)

    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix.png"),
                   f"ALL-data Eval | agreement={acc:.3f} kappa={kap:.3f}",
                   normalize=False)
    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix_norm.png"),
                   f"ALL-data Eval | agreement={acc:.3f} kappa={kap:.3f}",
                   normalize=True)

    metrics = {
        "mode": "train_on_all_eval_on_all (overfit showcase)",
        "seed": args.seed,
        "ctx": args.ctx,
        "epochs": args.epochs,
        "agreement": acc,
        "bal_acc": bal,
        "macro_f1": mf1,
        "kappa": kap,
        "kappa_quadratic": kap_q,
        "counts": dict(Counter(true.tolist()))
    }
    save_json(metrics, os.path.join(args.out_dir, "metrics.json"))

    # save weights
    ckpt_path = os.path.join(args.out_dir, "model_all_overfit.pt")
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "stage_names": STAGE_NAMES
    }, ckpt_path)

    # save predictions
    out_csv = os.path.join(args.out_dir, "predictions_all.csv")
    pd = __import__("pandas").DataFrame({
        "true": true,
        "pred": pred
    })
    pd.to_csv(out_csv, index=False)

    print("\n[DONE] Overfit showcase")
    print(metrics)
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()