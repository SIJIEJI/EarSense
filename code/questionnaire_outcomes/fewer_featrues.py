#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from traitlets import default


# ---------- utils ----------
def encode_classes(y_raw):
    uniq = pd.unique(y_raw)
    try:
        uniq_sorted = sorted(uniq, key=lambda x: float(x))
    except Exception:
        uniq_sorted = sorted(uniq, key=lambda x: str(x))
    mapping = {v: i for i, v in enumerate(uniq_sorted)}
    inv = {i: v for v, i in mapping.items()}
    y = np.array([mapping[v] for v in y_raw], dtype=np.int64)
    return y, mapping, inv


def masked_take(x_tf, mask_t):
    # x_tf: (T, k), mask_t: (T,)
    m = mask_t.astype(bool)
    if m.sum() == 0:
        return None
    return x_tf[m]


def safe_stats(arr_2d):
    """
    arr_2d: (N, k) after masking. We pool all channels together by flattening.
    return: a vector of robust time-domain stats.
    """
    if arr_2d is None or arr_2d.size == 0:
        return np.zeros((14,), dtype=np.float32)

    v = arr_2d.reshape(-1).astype(np.float32)
    if v.size < 5:
        # still return something
        mu = float(np.mean(v)) if v.size else 0.0
        sd = float(np.std(v)) + 1e-6
        mn = float(np.min(v)) if v.size else 0.0
        mx = float(np.max(v)) if v.size else 0.0
        med = float(np.median(v)) if v.size else 0.0
        q10 = float(np.quantile(v, 0.10)) if v.size else 0.0
        q90 = float(np.quantile(v, 0.90)) if v.size else 0.0
        iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25)) if v.size else 0.0
        energy = float(np.mean(v * v)) if v.size else 0.0
        zcr = 0.0
        slope = 0.0
        mad = float(np.median(np.abs(v - med))) if v.size else 0.0
        return np.array([mu, sd, mn, mx, med, q10, q90, iqr, mad, energy, zcr, slope, float(v.size), 0.0], dtype=np.float32)

    mu = float(np.mean(v))
    sd = float(np.std(v)) + 1e-6
    mn = float(np.min(v))
    mx = float(np.max(v))
    med = float(np.median(v))
    q10 = float(np.quantile(v, 0.10))
    q90 = float(np.quantile(v, 0.90))
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(v - med)))
    energy = float(np.mean(v * v))

    # zero-crossing rate (rough)
    sgn = np.sign(v)
    sgn[sgn == 0] = 1
    zcr = float(np.mean(sgn[1:] != sgn[:-1]))

    # trend slope vs index (normalized)
    t = np.linspace(0, 1, num=v.size, dtype=np.float32)
    # slope = cov(t,v)/var(t)
    vt = v - mu
    tt = t - float(np.mean(t))
    slope = float((vt * tt).mean() / (tt * tt).mean())

    # fraction of outliers beyond 3 sigma
    out3 = float(np.mean(np.abs(v - mu) > 3.0 * sd))

    return np.array([mu, sd, mn, mx, med, q10, q90, iqr, mad, energy, zcr, slope, float(v.size), out3], dtype=np.float32)


def extract_subject_features(X, mask, mod_map, segments=("full", "first_half", "second_half"), add_acc_mag=True):
    """
    X: (T,F), mask: (T,), mod_map: {modality: [indices]}
    Return: feature vector for one subject
    """
    T, F = X.shape
    feats = []

    # segment indices
    def seg_slice(name):
        if name == "full":
            return slice(0, T)
        if name == "first_half":
            return slice(0, T // 2)
        if name == "second_half":
            return slice(T // 2, T)
        if name == "first_third":
            return slice(0, T // 3)
        if name == "mid_third":
            return slice(T // 3, 2 * T // 3)
        if name == "last_third":
            return slice(2 * T // 3, T)
        raise ValueError(f"Unknown segment: {name}")

    for seg in segments:
        sl = seg_slice(seg)
        Xm = X[sl]
        mm = mask[sl]

        for mod, idxs in mod_map.items():
            idxs = [int(i) for i in idxs]
            if len(idxs) == 0:
                continue
            arr = masked_take(Xm[:, idxs], mm)
            feats.append(safe_stats(arr))

            # Optional: ACC magnitude feature if ACC has 3 axes
            if add_acc_mag and mod.upper() == "ACC" and len(idxs) >= 3:
                acc = Xm[:, idxs[:3]]
                if mm.astype(bool).sum() > 0:
                    acc = acc[mm.astype(bool)]
                    mag = np.sqrt((acc ** 2).sum(axis=1, keepdims=True))
                    feats.append(safe_stats(mag))
                else:
                    feats.append(np.zeros((14,), dtype=np.float32))

    return np.concatenate(feats, axis=0).astype(np.float32)


def plot_confusion(cm, labels, title, out_path, normalize=True):
    cm = np.array(cm, dtype=np.float32)
    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        cm = cm / rs

    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(cm, vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=25, ha="right")
    plt.yticks(ticks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                     ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", required=False, default="sleep_questionnaires_scores_one_row_per_subject_class.xlsx", help="xlsx file containing classification labels")
    ap.add_argument("--id_col", default="subject")
    ap.add_argument("--task_col", required=False, default="FOSQ_class", help="excel里你要分类的那一列，比如 'SSS_class'")
    ap.add_argument("--mod_json", required=False, default="modalities-EEGonly.json")
    ap.add_argument("--out_dir", default="./ml_summary_out")

    ap.add_argument("--segments", default="full,first_half,second_half",
                    help="可选: full,first_half,second_half,first_third,mid_third,last_third")
    ap.add_argument("--model", choices=["logreg", "svm", "rf", "hgb"], default="logreg")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize_cm", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load data
    d = np.load(args.npz, allow_pickle=True)
    X_all = d["X"].astype(np.float32)        # (N,T,F)
    mask_all = d["mask"].astype(np.float32)  # (N,T)
    subject_ids = d["subject_ids"].astype(str)

    # load labels
    df = pd.read_excel(args.label_xlsx)
    df[args.id_col] = df[args.id_col].astype(str)
    if args.task_col not in df.columns:
        raise ValueError(f"task_col '{args.task_col}' not in xlsx columns: {list(df.columns)}")
    df = df.dropna(subset=[args.task_col])
    label_map = df.set_index(args.id_col)

    keep = []
    y_raw = []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep.append(i)
            y_raw.append(label_map.loc[sid, args.task_col])

    keep = np.array(keep, dtype=np.int64)
    if len(keep) < 8:
        raise ValueError(f"Overlapping subjects too few: {len(keep)}. Check id alignment.")

    X_all = X_all[keep]
    mask_all = mask_all[keep]
    subject_ids = subject_ids[keep]
    y_raw = np.array(y_raw)

    y, mapping, inv = encode_classes(y_raw)
    n_classes = int(y.max() + 1)

    # modality mapping
    with open(args.mod_json, "r", encoding="utf-8") as f:
        mod_map = json.load(f)

    segs = [s.strip() for s in args.segments.split(",") if s.strip()]

    # extract features per subject
    feats = []
    for i in range(X_all.shape[0]):
        feats.append(extract_subject_features(X_all[i], mask_all[i], mod_map, segments=tuple(segs)))
    X_feat = np.stack(feats, axis=0)  # (N, D)

    # choose model
    if args.model == "logreg":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            multi_class="auto"
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif args.model == "svm":
        clf = SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=False
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced",
            random_state=args.seed
        )
        pipe = Pipeline([("clf", clf)])
    else:  # hgb
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=500,
            random_state=args.seed
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # stratified kfold (auto adjust k)
    counts = np.bincount(y, minlength=n_classes)
    k_use = min(args.k, int(counts.min()))
    if k_use < 2:
        raise ValueError(f"Some class has <2 samples; can't do stratified CV. counts={counts.tolist()}")

    skf = StratifiedKFold(n_splits=k_use, shuffle=True, random_state=args.seed)

    oof_pred = np.zeros_like(y)
    for fold, (tr, te) in enumerate(skf.split(X_feat, y), start=1):
        pipe.fit(X_feat[tr], y[tr])
        oof_pred[te] = pipe.predict(X_feat[te])
        acc = accuracy_score(y[te], oof_pred[te])
        bal = balanced_accuracy_score(y[te], oof_pred[te])
        mf1 = f1_score(y[te], oof_pred[te], average="macro")
        print(f"[Fold {fold}/{k_use}] acc={acc:.3f} bal_acc={bal:.3f} macro_f1={mf1:.3f} n_test={len(te)}")

    # overall
    acc = float(accuracy_score(y, oof_pred))
    bal = float(balanced_accuracy_score(y, oof_pred))
    mf1 = float(f1_score(y, oof_pred, average="macro"))
    cm = confusion_matrix(y, oof_pred, labels=list(range(n_classes)))

    # save
    out_csv = os.path.join(args.out_dir, "predictions_oof.csv")
    pd.DataFrame({
        "subject_id": subject_ids,
        "true": y,
        "pred": oof_pred
    }).to_csv(out_csv, index=False)

    labels_show = [str(inv[i]) for i in range(n_classes)]
    cm_png = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_confusion(cm, labels_show, f"{args.task_col} | acc={acc:.3f} bal={bal:.3f} mf1={mf1:.3f}",
                   cm_png, normalize=args.normalize_cm)

    summary = {
        "task_col": args.task_col,
        "model": args.model,
        "segments": segs,
        "n_subjects": int(len(y)),
        "n_features": int(X_feat.shape[1]),
        "class_counts": counts.tolist(),
        "acc": acc,
        "bal_acc": bal,
        "macro_f1": mf1,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[Overall]")
    print(summary)
    print("Saved:", out_csv)
    print("Saved:", cm_png)


if __name__ == "__main__":
    main()
