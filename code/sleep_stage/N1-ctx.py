#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced CNN Cascade + Context (ctx) for Sleep Staging

Labels:
  0=W, 1=N1, 2=N2, 3=N3, 4=REM

Stage-1 (binary): N1 vs Rest
Stage-2 (4-class): W / N2 / N3 / REM   (trained on non-N1)

ctx:
  ctx=1: use current epoch only
  ctx=3: use [t-1, t, t+1] concatenated on time axis  -> (C, 3T)
  ctx=5: use [t-2..t+2] -> (C, 5T)

Requirements:
  data_dir must contain:
    X.npy : (N,C,T) float32
    y.npy : (N,) int64 in {0..4}
  For ctx>1, strongly recommended:
    sid.npy : (N,) int64 subject id (ensures context doesn't cross subject boundary)

Run (recommended):
  python train_cascade_ctx_enhanced_cnn.py \
    --data_dir "./sleep_dataset_built/merged" \
    --out_dir "./cascade_ctx3_soft" \
    --ctx 3 \
    --cascade_mode soft \
    --alpha_n1 1.4

Tips:
  - If N2 is being sucked into N1: increase --alpha_n1 (e.g., 1.4 -> 1.6)
  - If N1 recall is too low: decrease --alpha_n1 (e.g., 1.4 -> 1.2)
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
DEVICE_AUTO = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# basic utils
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def split_70_30(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    try:
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    except ValueError:
        tr, te = train_test_split(idx, test_size=0.30, random_state=seed, shuffle=True)
    return tr, te

def plot_confusion(cm: np.ndarray, out_path: str, title: str, normalize: bool):
    if normalize:
        cm_disp = cm.astype(np.float32)
        cm_disp = cm_disp / (cm_disp.sum(axis=1, keepdims=True) + 1e-12)
    else:
        cm_disp = cm

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_disp)
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks(range(5), STAGE_NAMES)
    plt.yticks(range(5), STAGE_NAMES)

    for i in range(5):
        for j in range(5):
            txt = f"{cm_disp[i, j]*100:.1f}%" if normalize else str(int(cm_disp[i, j]))
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# augmentation (1D)
# -------------------------
def aug_time_shift(x, max_shift=80):
    if max_shift <= 0:
        return x
    s = np.random.randint(-max_shift, max_shift + 1)
    if s == 0:
        return x
    return np.roll(x, shift=s, axis=1)

def aug_scale(x, scale_min=0.9, scale_max=1.1):
    a = np.random.uniform(scale_min, scale_max)
    return x * a

def aug_jitter(x, sigma=0.01):
    return x + np.random.randn(*x.shape).astype(np.float32) * sigma

def aug_time_mask(x, max_masks=2, max_width=250):
    C, T = x.shape
    m = np.random.randint(0, max_masks + 1)
    for _ in range(m):
        w = np.random.randint(10, max_width + 1)
        s = np.random.randint(0, max(1, T - w))
        x[:, s:s+w] = 0.0
    return x

def apply_augment(x, p=0.7):
    if np.random.rand() > p:
        return x
    if np.random.rand() < 0.6:
        x = aug_time_shift(x, max_shift=80)
    if np.random.rand() < 0.6:
        x = aug_scale(x, 0.9, 1.1)
    if np.random.rand() < 0.6:
        x = aug_jitter(x, sigma=0.01)
    if np.random.rand() < 0.5:
        x = aug_time_mask(x, max_masks=2, max_width=250)
    return x


# -------------------------
# Context Dataset
# -------------------------
class ContextDataset(Dataset):
    """
    Returns x shape (C, ctx*T) by concatenating neighbor epochs within same subject.

    Requires sid for ctx>1 (recommended). If sid is None:
      it will still do context using global indices (may cross subjects) -> not recommended.
    """
    def __init__(self, X, y, sid, indices, ctx=1, augment=False):
        assert ctx % 2 == 1 and ctx >= 1
        self.X = X
        self.y = y
        self.sid = sid
        self.idx = np.array(indices, dtype=np.int64)
        self.ctx = ctx
        self.half = ctx // 2
        self.augment = augment

        self.group = None
        self.pos = None

        if self.sid is not None:
            self.group = {}
            for s in np.unique(self.sid):
                self.group[s] = np.where(self.sid == s)[0]  # assumes time-ordered within subject

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
                # fallback: global neighbors (not recommended)
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

        if self.augment:
            x = apply_augment(x.copy())

        return torch.tensor(x, dtype=torch.float32), int(self.y[gidx])


# -------------------------
# Loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, reduction="none",
            weight=self.alpha, label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# -------------------------
# Model: Enhanced ResNet1D + SE + dilation
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Conv1d(ch, mid, kernel_size=1)
        self.fc2 = nn.Conv1d(mid, ch, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch, r=8)
        self.drop = nn.Dropout(dropout)

        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
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
    def __init__(self, in_ch: int, num_classes: int, base=32, dropout=0.15):
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
        self.stage4 = nn.Sequential(
            ResBlock1D(base*4, base*4, k=5, stride=1, dilation=8, dropout=dropout),
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
        x = self.stage4(x)
        x = self.pool(x)
        x = self.head(x)
        return x


# -------------------------
# sampling helpers
# -------------------------
def sampler_invfreq(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w_class = 1.0 / counts
    w_sample = w_class[y]
    return WeightedRandomSampler(torch.tensor(w_sample, dtype=torch.double),
                                 num_samples=len(y), replacement=True)

def sampler_invsqrt(y, num_classes):
    # gentler balance: 1/sqrt(count)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w_class = 1.0 / np.sqrt(counts)
    w_sample = w_class[y]
    return WeightedRandomSampler(torch.tensor(w_sample, dtype=torch.double),
                                 num_samples=len(y), replacement=True)


# -------------------------
# train / eval
# -------------------------
@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        all_p.append(pred.cpu().numpy())
        all_t.append(yb.cpu().numpy())
    p = np.concatenate(all_p) if all_p else np.array([], dtype=int)
    t = np.concatenate(all_t) if all_t else np.array([], dtype=int)
    if len(t) == 0:
        return {"macro_f1": 0.0}
    return {
        "acc": float(accuracy_score(t, p)),
        "bal_acc": float(balanced_accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro")),
        "pred": p,
        "true": t
    }

def train_stage(
    X, y_stage, sid,
    train_indices, out_ckpt,
    ctx, seed,
    epochs, batch_size, lr,
    num_classes,
    device="auto",
    augment=True,
    focal=True,
    label_smoothing=0.05,
    sampler_kind="invfreq",   # "invfreq" or "invsqrt"
    patience=8
):
    set_seed(seed)
    dev = DEVICE_AUTO if device == "auto" else device

    # split train_indices into train/val
    idx = np.array(train_indices, dtype=np.int64)
    y_for_split = y_stage[idx]
    try:
        tr_idx, va_idx = train_test_split(idx, test_size=0.12, random_state=seed, stratify=y_for_split)
    except ValueError:
        tr_idx, va_idx = train_test_split(idx, test_size=0.12, random_state=seed, shuffle=True)

    ds_tr = ContextDataset(X, y_stage, sid, tr_idx, ctx=ctx, augment=augment)
    ds_va = ContextDataset(X, y_stage, sid, va_idx, ctx=ctx, augment=False)

    if sampler_kind == "invsqrt":
        sampler = sampler_invsqrt(y_stage[tr_idx], num_classes=num_classes)
    else:
        sampler = sampler_invfreq(y_stage[tr_idx], num_classes=num_classes)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size*2, shuffle=False, drop_last=False)

    model = EnhancedSleepCNN(in_ch=X.shape[1], num_classes=num_classes, base=32, dropout=0.15).to(dev)

    # class weights
    counts = np.bincount(y_stage[tr_idx], minlength=num_classes).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.mean()
    alpha = torch.tensor(w, dtype=torch.float32, device=dev)

    if focal:
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_f1 = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(dev)
            yb = yb.to(dev)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        va = eval_model(model, dl_va, dev)
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"[ep {ep:02d}] val macroF1={va['macro_f1']:.4f} acc={va['acc']:.4f} bal={va['bal_acc']:.4f} bad={bad}/{patience}")
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_ckpt)

    return model, dev, best_f1


@torch.no_grad()
def predict_proba(model, X, y_dummy, sid, indices, ctx, device, batch_size=256, num_classes=2):
    model.eval()
    ds = ContextDataset(X, y_dummy, sid, indices, ctx=ctx, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    probs = []
    for xb, _ in dl:
        xb = xb.to(device)
        logits = model(xb)
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


# -------------------------
# cascade inference
# -------------------------
def cascade_soft(p_n1, p_rest4, alpha_n1=1.4):
    """
    Soft cascade with N1 suppression:
      p_n1_adj = p_n1^alpha_n1  (alpha>1 reduces N1 false positives -> improves N2)
      P(N1)=p_n1_adj
      P(other)=(1-p_n1_adj)*p_rest4
    p_rest4 is over [W, N2, N3, REM] => map to [0,2,3,4]
    """
    p = np.clip(p_n1, 1e-6, 1 - 1e-6)
    p_adj = p ** alpha_n1
    scale = (1.0 - p_adj).astype(np.float32)

    N = len(p_n1)
    P5 = np.zeros((N, 5), dtype=np.float32)
    P5[:, 1] = p_adj
    P5[:, 0] = scale * p_rest4[:, 0]
    P5[:, 2] = scale * p_rest4[:, 1]
    P5[:, 3] = scale * p_rest4[:, 2]
    P5[:, 4] = scale * p_rest4[:, 3]
    y_pred = P5.argmax(axis=1).astype(np.int64)
    return y_pred, P5

def cascade_hard(p_n1, p_rest4, tau=0.5):
    y_pred = np.zeros((len(p_n1),), dtype=np.int64)
    is_n1 = p_n1 >= tau
    y_pred[is_n1] = 1
    idx = np.where(~is_n1)[0]
    if len(idx) > 0:
        pr = p_rest4[idx].argmax(axis=1)  # 0..3 => W,N2,N3,REM
        mapped = np.zeros_like(pr)
        mapped[pr == 0] = 0
        mapped[pr == 1] = 2
        mapped[pr == 2] = 3
        mapped[pr == 3] = 4
        y_pred[idx] = mapped
    return y_pred


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=False, default="sleep_dataset_built/merged", help="Folder containing X.npy and y.npy (and optional sid.npy)")
    ap.add_argument("--out_dir", default="./cascade_ctx_out")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ctx", type=int, default=3, help="1/3/5 recommended; ctx=3 uses neighbors [t-1,t,t+1]")
    ap.add_argument("--epochs1", type=int, default=200)
    ap.add_argument("--epochs2", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32, help="ctx>1 increases memory; use 16/32 if needed")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")

    ap.add_argument("--cascade_mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--alpha_n1", type=float, default=1.4, help="soft mode only: >1 reduces N1 false positives")
    ap.add_argument("--tau_n1", type=float, default=0.5, help="hard mode only")

    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--no_focal", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    X = np.load(os.path.join(args.data_dir, "X.npy")).astype(np.float32)  # (N,C,T)
    y = np.load(os.path.join(args.data_dir, "y.npy")).astype(np.int64)    # (N,)
    valid = (y >= 0) & (y <= 4)
    X, y = X[valid], y[valid]

    sid_path = os.path.join(args.data_dir, "sid.npy")
    sid = np.load(sid_path).astype(np.int64)[valid] if os.path.exists(sid_path) else None

    if args.ctx > 1 and sid is None:
        print("[WARN] sid.npy not found. ctx>1 will use global neighbors and may cross subjects (not recommended).")

    tr_idx, te_idx = split_70_30(y, seed=args.seed)
    print("[INFO] X:", X.shape, "y:", y.shape)
    print("[INFO] train counts:", dict(Counter(y[tr_idx].tolist())))
    print("[INFO] test  counts:", dict(Counter(y[te_idx].tolist())))

    # -------------------------
    # Stage-1: N1 vs Rest (binary)
    # -------------------------
    y1 = (y == 1).astype(np.int64)  # 1=N1, 0=rest
    print("\n[STAGE-1] Train N1 vs Rest (binary) ...")
    m1, dev1, best1 = train_stage(
        X=X, y_stage=y1, sid=sid,
        train_indices=tr_idx,
        out_ckpt=os.path.join(args.out_dir, "stage1_n1_vs_rest.pt"),
        ctx=args.ctx, seed=args.seed,
        epochs=args.epochs1, batch_size=args.batch_size, lr=args.lr,
        num_classes=2,
        device=args.device,
        augment=(not args.no_aug),
        focal=(not args.no_focal),
        label_smoothing=0.03,
        sampler_kind="invsqrt",   # key: gentler to reduce N2->N1
        patience=7
    )
    print("[STAGE-1] best val macroF1:", best1)

    # stage1 probs on test
    dummy = np.zeros((len(y),), dtype=np.int64)
    p1 = predict_proba(m1, X, dummy, sid, te_idx, ctx=args.ctx, device=dev1,
                       batch_size=max(128, args.batch_size*2), num_classes=2)
    p_n1 = p1[:, 1]  # P(N1)

    # -------------------------
    # Stage-2: 4-class on non-N1 (W/N2/N3/REM)
    # -------------------------
    # training indices exclude N1
    tr2_idx = tr_idx[y[tr_idx] != 1]
    # create y2 array (same length as y) but only meaningful for non-N1
    y2 = np.full((len(y),), -1, dtype=np.int64)
    map_to4 = {0: 0, 2: 1, 3: 2, 4: 3}  # [W, N2, N3, REM]
    for k, v in map_to4.items():
        y2[y == k] = v

    print("\n[STAGE-2] Train 4-class on non-N1: W/N2/N3/REM ...")
    m2, dev2, best2 = train_stage(
        X=X, y_stage=y2, sid=sid,
        train_indices=tr2_idx,
        out_ckpt=os.path.join(args.out_dir, "stage2_rest4.pt"),
        ctx=args.ctx, seed=args.seed + 1,
        epochs=args.epochs2, batch_size=args.batch_size, lr=args.lr,
        num_classes=4,
        device=args.device,
        augment=(not args.no_aug),
        focal=(not args.no_focal),
        label_smoothing=0.05,
        sampler_kind="invfreq",
        patience=8
    )
    print("[STAGE-2] best val macroF1:", best2)

    p2 = predict_proba(m2, X, dummy, sid, te_idx, ctx=args.ctx, device=dev2,
                       batch_size=max(128, args.batch_size*2), num_classes=4)  # (Ntest,4) over [W,N2,N3,REM]

    # -------------------------
    # Cascade inference
    # -------------------------
    y_te = y[te_idx]
    if args.cascade_mode == "soft":
        y_pred, _ = cascade_soft(p_n1, p2, alpha_n1=args.alpha_n1)
    else:
        y_pred = cascade_hard(p_n1, p2, tau=args.tau_n1)
    

        # -------------------------
    # Save per-epoch test predictions (for subject-level hypnogram)
    # -------------------------
    save_dict = {
        "te_idx": np.asarray(te_idx, dtype=np.int64),
        "y_true": np.asarray(y_te, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
    }
    if sid is not None:
        save_dict["sid"] = np.asarray(sid[te_idx], dtype=np.int64)

    np.savez(os.path.join(args.out_dir, "test_epoch_predictions.npz"), **save_dict)
    print("[INFO] Saved:", os.path.join(args.out_dir, "test_epoch_predictions.npz"))

    # -------------------------
    # Plot one subject hypnogram (PSG vs Patch) like your example
    # -------------------------
    from sklearn.metrics import cohen_kappa_score

    def _to_plot_y(y_5cls):
        """
        Map {0=W,1=N1,2=N2,3=N3,4=REM} -> y-axis order [W, R, N1, N2, N3]
        Returns values in {0..4} where 1 corresponds to R (=REM).
        """
        y_5cls = np.asarray(y_5cls, dtype=np.int64)
        yplot = np.empty_like(y_5cls)
        yplot[y_5cls == 0] = 0      # W
        yplot[y_5cls == 4] = 1      # REM -> R
        yplot[y_5cls == 1] = 2      # N1
        yplot[y_5cls == 2] = 3      # N2
        yplot[y_5cls == 3] = 4      # N3
        return yplot

    def plot_subject_hypnogram(y_true_all, y_pred_all, te_idx_all, sid_te, subject_id,
                               epoch_sec=30, out_png=None):
        # filter subject
        mask = (sid_te == subject_id)
        if mask.sum() == 0:
            raise ValueError(f"No test epochs for subject_id={subject_id}")

        sub_te = te_idx_all[mask]
        order = np.argsort(sub_te)  # ensure time order
        sub_true = y_true_all[mask][order]
        sub_pred = y_pred_all[mask][order]

        acc = float(accuracy_score(sub_true, sub_pred))
        kappa = float(cohen_kappa_score(sub_true, sub_pred, labels=[0,1,2,3,4]))

        t_min = (np.arange(len(sub_true)) * epoch_sec) / 60.0
        y_psg = _to_plot_y(sub_true)
        y_patch = _to_plot_y(sub_pred)

        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
        ax.step(t_min, y_psg, where="post", linewidth=1.2, label="PSG")

        offset = 6.0
        ax.step(t_min, y_patch + offset, where="post", linewidth=1.2, label="Patch")

        yticks = [0, 1, 2, 3, 4]
        ylabels = ["W", "R", "N1", "N2", "N3"]
        ax.set_yticks(yticks + [v + offset for v in yticks])
        ax.set_yticklabels(ylabels + ylabels)

        ax.axhline(5.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time (min)")
        ax.set_title(f"{acc*100:.2f}% agreement, κ = {kappa:.2f} (subject={subject_id})")
        ax.invert_yaxis()
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()

        if out_png is not None:
            plt.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print("[INFO] Saved:", out_png)
        else:
            plt.show()

    if sid is None:
        print("[WARN] sid.npy not found; skip subject hypnogram plotting.")
    else:
        sid_te = sid[te_idx].astype(np.int64)

        # choose subject with most test epochs (usually a full-night record)
        uniq, cnt = np.unique(sid_te, return_counts=True)
        best_sub = int(uniq[np.argmax(cnt)])

        out_png = os.path.join(args.out_dir, f"hypnogram_subject_{best_sub}.png")
        plot_subject_hypnogram(
            y_true_all=y_te,
            y_pred_all=y_pred,
            te_idx_all=np.asarray(te_idx, dtype=np.int64),
            sid_te=sid_te,
            subject_id=best_sub,
            epoch_sec=30,
            out_png=out_png
        )


    # -------------------------
    # Evaluation & save
    # -------------------------
    metrics = {
        "seed": args.seed,
        "ctx": args.ctx,
        "cascade_mode": args.cascade_mode,
        "alpha_n1": float(args.alpha_n1),
        "tau_n1": float(args.tau_n1),
        "stage1_best_val_macro_f1": float(best1),
        "stage2_best_val_macro_f1": float(best2),
        "test_acc": float(accuracy_score(y_te, y_pred)),
        "test_bal_acc": float(balanced_accuracy_score(y_te, y_pred)),
        "test_macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "train_counts": dict(Counter(y[tr_idx].tolist())),
        "test_counts": dict(Counter(y_te.tolist())),
    }
    save_json(metrics, os.path.join(args.out_dir, "metrics.json"))

    rep = classification_report(y_te, y_pred, labels=[0,1,2,3,4], target_names=STAGE_NAMES, zero_division=0)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(rep)

    cm = confusion_matrix(y_te, y_pred, labels=[0,1,2,3,4])
    np.savetxt(os.path.join(args.out_dir, "confusion_counts.txt"), cm, fmt="%d")

    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix.png"),
                   "Cascade+CTX Enhanced CNN Confusion Matrix (Test)", normalize=False)
    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix_norm.png"),
                   "Cascade+CTX Enhanced CNN Confusion Matrix (Test)", normalize=True)

    print("\n[DONE] Cascade+CTX Enhanced CNN")
    print(json.dumps(metrics, indent=2))
    print("\nSaved to:", args.out_dir)


if __name__ == "__main__":
    main()
