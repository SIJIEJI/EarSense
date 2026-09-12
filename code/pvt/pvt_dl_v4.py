#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PVT Train-on-All + Simple MLP + 5 batch-subset tests

Goal:
- Train on ALL data (no CV, no split)
- Test on 5 different batch subsets (each subset is one "batch" of samples)
- Output per-batch metrics + summary CSV

Based on: /mnt/data/pvt_overfit.py
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
# Feature engineering (keep your existing feature builder)
# ─────────────────────────────────────────────────────

def masked_mean_std(X, mask):
    m   = mask.astype(bool)[..., None]
    cnt = m.sum(axis=1).clip(min=1)
    mu  = (X * m).sum(axis=1) / cnt
    mu2 = ((X**2) * m).sum(axis=1) / cnt
    var = np.maximum(mu2 - mu**2, 1e-6)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)

def build_features(X_raw, mask):
    """
    Keep it consistent with your original file for comparability:
    original mean/std (66 dims) + square (66 dims) + row stats (4 dims)
    """
    mu, sd = masked_mean_std(X_raw, mask)
    tab = np.concatenate([mu, sd], axis=1).astype(np.float64)   # (N, 66)
    tab_sq = tab ** 2                                           # (N, 66)

    row_mean  = tab.mean(1, keepdims=True)
    row_std   = tab.std(1,  keepdims=True) + 1e-8
    row_max   = tab.max(1,  keepdims=True)
    row_min   = tab.min(1,  keepdims=True)

    feat = np.concatenate([tab, tab_sq, row_mean, row_std, row_max, row_min], axis=1)
    feat = np.where(np.isfinite(feat), feat, 0.0)
    return feat.astype(np.float64)

# ─────────────────────────────────────────────────────
# Simple Numpy MLP (much simpler than your overfit net)
# - 1~2 hidden layers
# - no fancy LR schedule
# - still uses Adam
# ─────────────────────────────────────────────────────

class SimpleMLP:
    def __init__(self, in_dim: int, hidden=(64,), seed: int = 0):
        np.random.seed(seed)
        dims = [in_dim] + list(hidden) + [1]
        self.Ws = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
                   for i in range(len(dims)-1)]
        self.bs = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]

    def _fwd(self, X):
        acts = [X]
        h = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.maximum(0.0, h @ W + b)   # ReLU
            acts.append(h)
        out = (h @ self.Ws[-1] + self.bs[-1]).ravel()
        return out, acts

    def _bwd(self, acts, pred, y):
        N = len(y)
        dL = 2.0 * (pred - y) / N

        dWs = [None] * len(self.Ws)
        dbs = [None] * len(self.bs)

        # last layer
        dWs[-1] = acts[-1].T @ dL.reshape(-1, 1)
        dbs[-1] = np.array([dL.sum()]) if self.bs[-1].ndim == 1 and self.bs[-1].shape[0] == 1 else dL.sum(keepdims=True)

        d = dL.reshape(-1, 1) @ self.Ws[-1].T
        for i in range(len(self.Ws)-2, -1, -1):
            d = d * (acts[i+1] > 0)
            dWs[i] = acts[i].T @ d
            dbs[i] = d.sum(axis=0)
            d = d @ self.Ws[i].T

        return dWs, dbs

    def fit(self, X, y, epochs=50, lr=1e-2, verbose_every=500):
        # normalize inside
        self.mu_x = X.mean(0)
        self.sd_x = X.std(0) + 1e-8
        self.mu_y = float(y.mean())
        self.sd_y = float(y.std()) + 1e-8

        Xs = (X - self.mu_x) / self.sd_x
        ys = (y - self.mu_y) / self.sd_y

        # Adam states
        mW = [np.zeros_like(w) for w in self.Ws]
        vW = [np.zeros_like(w) for w in self.Ws]
        mb = [np.zeros_like(b) for b in self.bs]
        vb = [np.zeros_like(b) for b in self.bs]
        b1, b2, eps = 0.9, 0.999, 1e-8

        self.loss_curve = []

        for ep in range(1, epochs + 1):
            pred, acts = self._fwd(Xs)
            loss = float(np.mean((pred - ys) ** 2))
            self.loss_curve.append(loss)

            dWs, dbs = self._bwd(acts, pred, ys)

            for i in range(len(self.Ws)):
                mW[i] = b1 * mW[i] + (1 - b1) * dWs[i]
                vW[i] = b2 * vW[i] + (1 - b2) * (dWs[i] ** 2)
                mb[i] = b1 * mb[i] + (1 - b1) * dbs[i]
                vb[i] = b2 * vb[i] + (1 - b2) * (dbs[i] ** 2)

                mWh = mW[i] / (1 - b1 ** ep)
                vWh = vW[i] / (1 - b2 ** ep)
                mbh = mb[i] / (1 - b1 ** ep)
                vbh = vb[i] / (1 - b2 ** ep)

                self.Ws[i] -= lr * mWh / (np.sqrt(vWh) + eps)
                self.bs[i] -= lr * mbh / (np.sqrt(vbh) + eps)

            if verbose_every and ep % verbose_every == 0:
                pred_orig = pred * self.sd_y + self.mu_y
                y_orig = ys * self.sd_y + self.mu_y
                mae = mean_absolute_error(y_orig, pred_orig)
                r2 = r2_score(y_orig, pred_orig)
                print(f"  ep {ep:>6d}  loss={loss:.6f}  MAE={mae:.3f}  R²={r2:.4f}")

        return self

    def predict(self, X):
        Xs = (X - self.mu_x) / self.sd_x
        pred, _ = self._fwd(Xs)
        return pred * self.sd_y + self.mu_y

    def save(self, path):
        """
        Save model weights and normalization parameters to .npz
        """
        if not hasattr(self, "mu_x"):
            raise RuntimeError("Model is not fitted yet. Please call fit() before save().")

        np.savez_compressed(
            path,
            Ws=np.array(self.Ws, dtype=object),
            bs=np.array(self.bs, dtype=object),
            mu_x=self.mu_x,
            sd_x=self.sd_x,
            mu_y=np.array([self.mu_y], dtype=np.float64),
            sd_y=np.array([self.sd_y], dtype=np.float64),
            hidden=np.array([w.shape[1] for w in self.Ws[:-1]], dtype=np.int64),
            in_dim=np.array([self.Ws[0].shape[0]], dtype=np.int64),
        )

    @classmethod
    def load(cls, path):
        """
        Load model weights and normalization parameters from .npz
        """
        ckpt = np.load(path, allow_pickle=True)

        hidden = tuple(ckpt["hidden"].astype(int).tolist())
        in_dim = int(ckpt["in_dim"][0])

        net = cls(in_dim=in_dim, hidden=hidden, seed=0)

        net.Ws = [ckpt["Ws"][i] for i in range(len(ckpt["Ws"]))]
        net.bs = [ckpt["bs"][i] for i in range(len(ckpt["bs"]))]

        net.mu_x = ckpt["mu_x"]
        net.sd_x = ckpt["sd_x"]
        net.mu_y = float(ckpt["mu_y"][0])
        net.sd_y = float(ckpt["sd_y"][0])

        return net

# class SimpleMLP:
#     def __init__(self, in_dim: int, hidden=(64,), seed: int = 0):
#         np.random.seed(seed)
#         dims = [in_dim] + list(hidden) + [1]
#         self.Ws = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
#                    for i in range(len(dims)-1)]
#         self.bs = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]

#     def _fwd(self, X):
#         acts = [X]
#         h = X
#         for W, b in zip(self.Ws[:-1], self.bs[:-1]):
#             h = np.maximum(0.0, h @ W + b)   # ReLU
#             acts.append(h)
#         out = (h @ self.Ws[-1] + self.bs[-1]).ravel()
#         return out, acts

#     def _bwd(self, acts, pred, y):
#         N = len(y)
#         dL = 2.0 * (pred - y) / N

#         dWs = [None] * len(self.Ws)
#         dbs = [None] * len(self.bs)

#         # last layer
#         dWs[-1] = acts[-1].T @ dL.reshape(-1, 1)
#         dbs[-1] = dL.sum(keepdims=True)

#         d = dL.reshape(-1, 1) @ self.Ws[-1].T
#         for i in range(len(self.Ws)-2, -1, -1):
#             d = d * (acts[i+1] > 0)
#             dWs[i] = acts[i].T @ d
#             dbs[i] = d.sum(axis=0)
#             d = d @ self.Ws[i].T

#         return dWs, dbs

#     def fit(self, X, y, epochs=50, lr=1e-2, verbose_every=500):
#         # normalize inside
#         self.mu_x = X.mean(0); self.sd_x = X.std(0) + 1e-8
#         self.mu_y = float(y.mean()); self.sd_y = float(y.std()) + 1e-8
#         Xs = (X - self.mu_x) / self.sd_x
#         ys = (y - self.mu_y) / self.sd_y

#         # Adam states
#         mW = [np.zeros_like(w) for w in self.Ws]; vW = [np.zeros_like(w) for w in self.Ws]
#         mb = [np.zeros_like(b) for b in self.bs]; vb = [np.zeros_like(b) for b in self.bs]
#         b1, b2, eps = 0.9, 0.999, 1e-8

#         self.loss_curve = []

#         for ep in range(1, epochs + 1):
#             pred, acts = self._fwd(Xs)
#             loss = float(np.mean((pred - ys) ** 2))
#             self.loss_curve.append(loss)

#             dWs, dbs = self._bwd(acts, pred, ys)

#             for i in range(len(self.Ws)):
#                 mW[i] = b1*mW[i] + (1-b1)*dWs[i]
#                 vW[i] = b2*vW[i] + (1-b2)*(dWs[i]**2)
#                 mb[i] = b1*mb[i] + (1-b1)*dbs[i]
#                 vb[i] = b2*vb[i] + (1-b2)*(dbs[i]**2)

#                 mWh = mW[i] / (1 - b1**ep); vWh = vW[i] / (1 - b2**ep)
#                 mbh = mb[i] / (1 - b1**ep); vbh = vb[i] / (1 - b2**ep)

#                 self.Ws[i] -= lr * mWh / (np.sqrt(vWh) + eps)
#                 self.bs[i] -= lr * mbh / (np.sqrt(vbh) + eps)

#             if verbose_every and ep % verbose_every == 0:
#                 pred_orig = pred * self.sd_y + self.mu_y
#                 y_orig = ys * self.sd_y + self.mu_y
#                 mae = mean_absolute_error(y_orig, pred_orig)
#                 r2 = r2_score(y_orig, pred_orig)
#                 print(f"  ep {ep:>6d}  loss={loss:.6f}  MAE={mae:.3f}  R²={r2:.4f}")

#         return self

#     def predict(self, X):
#         Xs = (X - self.mu_x) / self.sd_x
#         pred, _ = self._fwd(Xs)
#         return pred * self.sd_y + self.mu_y

# ─────────────────────────────────────────────────────
# Batch subset selection + evaluation
# ─────────────────────────────────────────────────────

def make_batches(N, batch_size, seed, shuffle=True):
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    if shuffle:
        rng.shuffle(idx)
    batches = [idx[i:i+batch_size] for i in range(0, N, batch_size)]
    return batches

def eval_subset(name, y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"subset": name, "n": int(len(y_true)), "MAE": mae, "RMSE": rmse, "R2": r2}

def plot_loss(loss_curve, out_path):
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_curve, lw=1.2)
    plt.xlabel("Epoch"); plt.ylabel("MSE (normalized)")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_parity(y_true, y_pred, title, out_path):
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], lw=1.2)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_residuals(y_true, y_pred, title, out_path):
    r = (y_pred - y_true)
    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(y_true, r, s=18, alpha=0.75)
    plt.axhline(0.0, lw=1.2)
    plt.xlabel("True")
    plt.ylabel("Residual (Pred - True)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_error_hist(y_true, y_pred, title, out_path, bins=20):
    ae = np.abs(y_pred - y_true)
    plt.figure(figsize=(6.2, 4.2))
    plt.hist(ae, bins=bins, alpha=0.9)
    plt.xlabel("|Pred - True|")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",        default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col",     default="subject")
    ap.add_argument("--target_col", default="mean_slowest_10pct_ms")
    ap.add_argument("--out_dir",    default="./pvt_mlp_all_train_out")

    # MLP config (simpler)
    ap.add_argument("--hidden", nargs="+", type=int, default=[64],
                    help="Simple MLP hidden sizes, e.g. 64 or 128 64")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose_every", type=int, default=500)

    # batch-subset test config
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Define what a 'batch subset' means (number of samples per batch).")
    ap.add_argument("--num_test_batches", type=int, default=5,
                    help="How many distinct batches to test on.")
    ap.add_argument("--batch_seed", type=int, default=123,
                    help="Seed for batch shuffling/selection.")
    ap.add_argument("--shuffle_batches", action="store_true",
                    help="Shuffle indices before batching (recommended).")
    ap.add_argument("--test_batch_ids", nargs="*", type=int, default=None,
                    help="Optional: explicitly choose which batch IDs to evaluate, e.g. 0 3 7 9 12.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load npz ──
    d = np.load(args.npz, allow_pickle=True)
    X_raw       = d["X"].astype(np.float32)
    mask        = d["mask"].astype(np.float32)
    subject_ids = d["subject_ids"].astype(str)

    # ── load labels ──
    df = pd.read_excel(args.label_xlsx)
    df[args.id_col] = df[args.id_col].astype(str)
    df = df.dropna(subset=[args.target_col])
    label_map = df.set_index(args.id_col)

    keep, y_list, kept_sids = [], [], []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep.append(i)
            y_list.append(float(label_map.loc[sid, args.target_col]))
            kept_sids.append(sid)

    keep = np.array(keep, dtype=np.int64)
    X_raw = X_raw[keep]; mask = mask[keep]
    subject_ids = np.array(kept_sids, dtype=str)
    y = np.array(y_list, dtype=np.float64)

    N = len(y)
    print(f"\n{'='*60}")
    print(f"Train-on-All + Simple MLP | N={N}")
    print(f"target: {args.target_col}")
    print(f"hidden: {args.hidden}, epochs={args.epochs}, lr={args.lr}")
    print(f"batch_size={args.batch_size}, num_test_batches={args.num_test_batches}")
    print(f"{'='*60}\n")

    # ── features ──
    Xtab = build_features(X_raw, mask)
    print(f"Feature matrix: {Xtab.shape}\n")

    # ── train on ALL ──
    net = SimpleMLP(in_dim=Xtab.shape[1], hidden=tuple(args.hidden), seed=args.seed)
    net.fit(Xtab, y, epochs=args.epochs, lr=args.lr, verbose_every=args.verbose_every)
    # ── save model weights ──
    model_path = os.path.join(args.out_dir, "simple_mlp_weights.npz")
    net.save(model_path)
    print(f"Saved model weights: {model_path}")

    # ── predictions on ALL ──
    yhat_all = net.predict(Xtab)

    # overall metrics (still “test on all” is not meaningful, but good sanity check)
    overall = eval_subset("ALL_DATA", y, yhat_all)
    print("\nOverall (same data used for training):")
    print(overall)

    # ── build batches and pick 5 to test ──
    batches = make_batches(
        N=N,
        batch_size=args.batch_size,
        seed=args.batch_seed,
        shuffle=args.shuffle_batches
    )
    num_batches = len(batches)
    print(f"\nTotal batches formed: {num_batches}")

    if args.test_batch_ids is not None and len(args.test_batch_ids) > 0:
        chosen = args.test_batch_ids[:args.num_test_batches]
    else:
        rng = np.random.RandomState(args.batch_seed + 999)
        chosen = rng.choice(np.arange(num_batches), size=min(args.num_test_batches, num_batches), replace=False).tolist()

    chosen = [int(b) for b in chosen]
    print(f"Chosen batch IDs for testing: {chosen}")

    # ── evaluate each chosen batch subset ──
    rows = [overall]
    pred_rows = []

    for bid in chosen:
        idx = batches[bid]
        y_true = y[idx]
        y_pred = yhat_all[idx]
        m = eval_subset(f"BATCH_{bid}", y_true, y_pred)
        rows.append(m)

        for k, i in enumerate(idx):
            pred_rows.append({
                "subset": f"BATCH_{bid}",
                "batch_id": bid,
                "subject_id": subject_ids[i],
                "true": float(y[i]),
                "pred": float(yhat_all[i]),
                "residual": float(yhat_all[i] - y[i]),
                "abs_error": float(abs(yhat_all[i] - y[i])),
            })

    # ── save outputs ──
    metrics_csv = os.path.join(args.out_dir, "metrics_by_subset.csv")
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics: {metrics_csv}")

    preds_csv = os.path.join(args.out_dir, "predictions_test_batches.csv")
    pd.DataFrame(pred_rows).to_csv(preds_csv, index=False)
    print(f"Saved batch predictions: {preds_csv}")

    loss_png = os.path.join(args.out_dir, "loss_curve.png")
    plot_loss(net.loss_curve, loss_png)
    print(f"Saved loss plot: {loss_png}")

    config = vars(args)
    config["N"] = int(N)
    config["num_batches"] = int(num_batches)
    config["chosen_batch_ids"] = chosen
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {os.path.join(args.out_dir, 'run_config.json')}\n")

        # ── also save full predictions for ALL samples ──
    all_csv = os.path.join(args.out_dir, "predictions_all_samples.csv")
    pd.DataFrame({
        "subject_id": subject_ids,
        "true": y.astype(float),
        "pred": yhat_all.astype(float),
        "residual": (yhat_all - y).astype(float),
        "abs_error": np.abs(yhat_all - y).astype(float),
    }).to_csv(all_csv, index=False)
    print(f"Saved all-sample predictions: {all_csv}")

    # ── plots for ALL_DATA ──
    plot_parity(y, yhat_all, "Parity Plot (ALL_DATA)", os.path.join(args.out_dir, "parity_all.png"))
    plot_residuals(y, yhat_all, "Residuals vs True (ALL_DATA)", os.path.join(args.out_dir, "residuals_all.png"))
    plot_error_hist(y, yhat_all, "Absolute Error Histogram (ALL_DATA)", os.path.join(args.out_dir, "abs_error_hist_all.png"))
    print("Saved plots: parity_all.png, residuals_all.png, abs_error_hist_all.png")

    # ── plots per chosen batch subset ──
    for bid in chosen:
        idx = batches[bid]
        y_true = y[idx]
        y_pred = yhat_all[idx]
        plot_parity(y_true, y_pred, f"Parity Plot (BATCH_{bid})",
                    os.path.join(args.out_dir, f"parity_batch_{bid}.png"))
        plot_residuals(y_true, y_pred, f"Residuals vs True (BATCH_{bid})",
                       os.path.join(args.out_dir, f"residuals_batch_{bid}.png"))
        plot_error_hist(y_true, y_pred, f"Abs Error Hist (BATCH_{bid})",
                        os.path.join(args.out_dir, f"abs_error_hist_batch_{bid}.png"))
    print("Saved per-batch plots: parity/residuals/error hist PNGs")

if __name__ == "__main__":
    main()