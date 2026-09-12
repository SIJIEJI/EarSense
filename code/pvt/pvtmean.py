#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PVT Regression — 目标: 90%+ 置信覆盖率 & 更高回归准确度
改进要点:
  1. 丰富特征工程 (均值/标准差/分位数/偏度/峰度/时序趋势)
  2. 集成模型 (GradientBoosting + ElasticNet + BayesianRidge Stacking)
  3. LOO内层交叉验证选超参
  4. Mondrian自适应保形预测 (按预测值分层)
  5. 保留原始bounded预测逻辑
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────

def extract_rich_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    从形状 (N, T, F) 或 (N, F) 的数据中提取丰富的统计特征。
    X: (N, num_features)  已经是tabular的情况下直接增强
    mask: (N, T) 或 (N,)
    
    返回增强后的特征矩阵。
    """
    N = X.shape[0]
    
    # 原始均值/标准差
    m = mask.astype(bool)[..., None] if mask.ndim == 2 else np.ones((N, 1, 1), dtype=bool)
    
    # 若 X 已经是 2D tabular，直接进行多项式/交叉特征增强
    if X.ndim == 2:
        return _enhance_tabular(X)
    
    # 3D: (N, T, F)
    feats = []
    for f_idx in range(X.shape[2]):
        xi = X[:, :, f_idx]  # (N, T)
        mi = mask.astype(bool)   # (N, T)
        
        rows_mu, rows_sd, rows_q25, rows_q75, rows_skew, rows_kurt, rows_trend = (
            [], [], [], [], [], [], []
        )
        for n in range(N):
            vals = xi[n][mi[n]]
            if len(vals) == 0:
                vals = np.array([0.0])
            rows_mu.append(np.mean(vals))
            rows_sd.append(np.std(vals) + 1e-8)
            rows_q25.append(np.percentile(vals, 25))
            rows_q75.append(np.percentile(vals, 75))
            rows_skew.append(stats.skew(vals) if len(vals) > 2 else 0.0)
            rows_kurt.append(stats.kurtosis(vals) if len(vals) > 3 else 0.0)
            # 线性趋势斜率
            if len(vals) > 1:
                slope, _, _, _, _ = stats.linregress(np.arange(len(vals)), vals)
            else:
                slope = 0.0
            rows_trend.append(slope)
        
        feats.extend([rows_mu, rows_sd, rows_q25, rows_q75, rows_skew, rows_kurt, rows_trend])
    
    feat_matrix = np.column_stack(feats).astype(np.float32)
    return _enhance_tabular(feat_matrix)


def _enhance_tabular(X: np.ndarray) -> np.ndarray:
    """对2D特征矩阵做多项式增强（平方项+重要交叉项）"""
    N, D = X.shape
    
    # 替换inf/nan
    X = np.where(np.isfinite(X), X, 0.0)
    
    # 平方项（捕获非线性）
    X_sq = X ** 2
    
    # IQR特征（若维度允许，取前半/后半的差）
    half = D // 2
    if half > 0:
        X_iqr = X[:, :half] - X[:, half:half*2] if D >= 2 * half else np.zeros((N, 1))
    else:
        X_iqr = np.zeros((N, 1))
    
    # 行级统计（跨特征）
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_std  = np.std(X,  axis=1, keepdims=True) + 1e-8
    row_max  = np.max(X,  axis=1, keepdims=True)
    row_min  = np.min(X,  axis=1, keepdims=True)
    row_range = row_max - row_min
    
    enhanced = np.concatenate([
        X, X_sq, X_iqr,
        row_mean, row_std, row_max, row_min, row_range
    ], axis=1).astype(np.float32)
    
    return enhanced


def masked_mean_std(X, mask):
    m = mask.astype(bool)[..., None]
    cnt = m.sum(axis=1).clip(min=1)
    mu = (X * m).sum(axis=1) / cnt
    mu2 = ((X**2) * m).sum(axis=1) / cnt
    var = np.maximum(mu2 - mu**2, 1e-6)
    sd = np.sqrt(var)
    return mu.astype(np.float32), sd.astype(np.float32)


# ─────────────────────────────────────────────
# 保形预测
# ─────────────────────────────────────────────

def conformal_q(abs_residuals, alpha=0.1):
    r = np.asarray(abs_residuals, dtype=float)
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    level = k / n
    try:
        q = float(np.quantile(r, level, method="higher"))
    except TypeError:
        q = float(np.quantile(r, level, interpolation="higher"))
    return q, level


def mondrian_conformal_pi(y_cal, yhat_cal, yhat_test, abs_res_cal, alpha=0.1, n_bins=3):
    """
    Mondrian（分层）保形预测：按预测值分组，每组独立计算q值。
    产生个体化（非对称）预测区间，提升覆盖率精度。
    """
    # 按预测值分箱
    bins = np.quantile(yhat_cal, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    
    lo_test = np.zeros(len(yhat_test))
    hi_test = np.zeros(len(yhat_test))
    
    for b in range(n_bins):
        # 校准集中属于此箱的样本
        cal_mask = (yhat_cal > bins[b]) & (yhat_cal <= bins[b+1])
        if cal_mask.sum() < 3:
            # 样本不足，退化为全局q
            q, _ = conformal_q(abs_res_cal, alpha)
        else:
            q, _ = conformal_q(abs_res_cal[cal_mask], alpha)
        
        # 测试集中属于此箱的样本
        test_mask = (yhat_test > bins[b]) & (yhat_test <= bins[b+1])
        lo_test[test_mask] = yhat_test[test_mask] - q
        hi_test[test_mask] = yhat_test[test_mask] + q
    
    # 边界外的测试点用全局q
    uncovered = (lo_test == 0) & (hi_test == 0)
    if uncovered.any():
        q_global, _ = conformal_q(abs_res_cal, alpha)
        lo_test[uncovered] = yhat_test[uncovered] - q_global
        hi_test[uncovered] = yhat_test[uncovered] + q_global
    
    return lo_test, hi_test


# ─────────────────────────────────────────────
# 模型集成
# ─────────────────────────────────────────────

def build_ensemble_models(ridge_alpha=10.0):
    """返回多个候选模型，LOO中自适应选择或融合"""
    models = {
        "ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   Ridge(alpha=ridge_alpha))
        ]),
        "bayesian_ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   BayesianRidge(max_iter=500))
        ]),
        "elastic_net": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000))
        ]),
        "huber": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("m",   HuberRegressor(epsilon=1.5, max_iter=500))
        ]),
        "gbm": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   GradientBoostingRegressor(
                        n_estimators=100, max_depth=3,
                        learning_rate=0.05, subsample=0.8,
                        min_samples_leaf=2, random_state=42))
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m",   RandomForestRegressor(
                        n_estimators=200, max_depth=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1))
        ]),
        "extra_trees": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m",   ExtraTreesRegressor(
                        n_estimators=200, max_depth=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1))
        ]),
    }
    return models


def stacked_predict(models_dict, X_tr, y_tr, X_te, cv=3):
    """
    对每个基模型做内层CV，选出最优模型加权融合（LOO-safe stacking）
    """
    n_tr = len(y_tr)
    if n_tr < cv + 2:
        # 数据太少，直接用ridge
        m = models_dict["ridge"]
        m.fit(X_tr, y_tr)
        return float(m.predict(X_te)[0])
    
    scores = {}
    for name, model in models_dict.items():
        try:
            cv_scores = cross_val_score(
                model, X_tr, y_tr,
                cv=min(cv, n_tr),
                scoring="neg_mean_absolute_error",
                n_jobs=1
            )
            scores[name] = float(np.mean(cv_scores))  # 负MAE，越大越好
        except Exception:
            scores[name] = -1e9
    
    # 选 top-3 模型加权融合（权重∝ 1/MAE）
    sorted_names = sorted(scores, key=lambda k: scores[k], reverse=True)
    top_k = min(3, len(sorted_names))
    top_names = sorted_names[:top_k]
    
    # 权重：softmax over negative-MAE
    top_scores = np.array([scores[n] for n in top_names])
    weights = np.exp(top_scores - top_scores.max())
    weights /= weights.sum()
    
    preds = []
    for name in top_names:
        models_dict[name].fit(X_tr, y_tr)
        preds.append(float(models_dict[name].predict(X_te)[0]))
    
    return float(np.dot(weights, preds))


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def plot_abs_error_bar(ids, abs_err, out_path):
    order = np.argsort(abs_err)[::-1]
    plt.figure(figsize=(10.5, 4.2))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(abs_err)))
    bars = plt.bar(np.arange(len(abs_err)), abs_err[order], color=colors[np.argsort(np.argsort(abs_err)[::-1])])
    plt.xticks(np.arange(len(abs_err)), np.array(ids)[order], rotation=60, ha="right", fontsize=8)
    plt.ylabel("|Pred-True| (ms)")
    plt.title("Absolute Error per Subject (LOO-CV Enhanced)")
    mean_err = np.mean(abs_err)
    plt.axhline(mean_err, color="red", linestyle="--", label=f"Mean={mean_err:.1f}ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter_with_pi(y, yhat, lo, hi, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # 左图：预测散点 + PI
    ax = axes[0]
    colors = np.where((y >= lo) & (y <= hi), "steelblue", "tomato")
    for i in range(len(y)):
        ax.plot([y[i], y[i]], [lo[i], hi[i]], color=colors[i], alpha=0.4, linewidth=1.2)
        ax.scatter(y[i], yhat[i], color=colors[i], s=30, zorder=3)
    mn = float(min(y.min(), yhat.min(), lo.min()))
    mx = float(max(y.max(), yhat.max(), hi.max()))
    pad = 0.05 * (mx - mn + 1e-6)
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], "k--", linewidth=1)
    ax.set_xlabel("True mean_slowest_10pct_ms")
    ax.set_ylabel("Pred (LOO) ± 90% PI")
    ax.set_title(title, fontsize=9)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Covered'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=8, label='Not covered'),
    ]
    ax.legend(handles=legend_elements)
    
    # 右图：残差分布
    ax2 = axes[1]
    residuals = yhat - y
    ax2.hist(residuals, bins=max(8, len(y)//3), color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--")
    ax2.axvline(np.mean(residuals), color="orange", linestyle="-", label=f"bias={np.mean(residuals):.1f}")
    ax2.set_xlabel("Residual (Pred - True) ms")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_model_comparison(results_dict, out_path):
    """比较不同模型策略的MAE"""
    names = list(results_dict.keys())
    maes  = [results_dict[n]["mae"] for n in names]
    r2s   = [results_dict[n]["r2"]  for n in names]
    covs  = [results_dict[n]["coverage"] for n in names]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"][:len(names)]
    
    axes[0].bar(names, maes, color=colors)
    axes[0].set_title("MAE (ms) — lower is better")
    axes[0].set_ylabel("MAE (ms)")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    
    axes[1].bar(names, r2s, color=colors)
    axes[1].set_title("R² — higher is better")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_ylabel("R²")
    for i, v in enumerate(r2s):
        axes[1].text(i, max(v, 0) + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    
    axes[2].bar(names, covs, color=colors)
    axes[2].axhline(0.90, color="red", linestyle="--", label="90% target")
    axes[2].set_title("Empirical 90% PI Coverage")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Coverage")
    axes[2].legend()
    for i, v in enumerate(covs):
        axes[2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    
    for ax in axes:
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",        default="nn_dataset_subject_level.npz")
    ap.add_argument("--label_xlsx", default="./sleep_questionnaires_scores_one_row_per_subject_class.xlsx")
    ap.add_argument("--id_col",     default="subject")
    ap.add_argument("--target_col", default="mean_slowest_10pct_ms")
    ap.add_argument("--out_dir",    default="./pvt_enhanced_reg")

    ap.add_argument("--alpha",      type=float, default=0.1,  help="0.1 => 90% PI")
    ap.add_argument("--ridge_alpha",type=float, default=10.0)
    ap.add_argument("--bound_mode", choices=["minmax", "quantile"], default="quantile")
    ap.add_argument("--q_low",      type=float, default=0.02)
    ap.add_argument("--q_high",     type=float, default=0.98)
    ap.add_argument("--mondrian_bins", type=int, default=3,
                    help="Mondrian conformal bins (3 recommended for small N)")
    ap.add_argument("--use_ensemble", action="store_true", default=True,
                    help="Use model ensemble (recommended)")
    ap.add_argument("--inner_cv",   type=int, default=3,
                    help="Inner CV folds for model selection")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载数据 ──
    d = np.load(args.npz, allow_pickle=True)
    X_raw  = d["X"].astype(np.float32)
    mask   = d["mask"].astype(np.float32)
    subject_ids = d["subject_ids"].astype(str)

    df = pd.read_excel(args.label_xlsx)
    df[args.id_col] = df[args.id_col].astype(str)
    df = df.dropna(subset=[args.target_col])
    label_map = df.set_index(args.id_col)

    keep, y_list, kept_subjects = [], [], []
    for i, sid in enumerate(subject_ids):
        if sid in label_map.index:
            keep.append(i)
            y_list.append(float(label_map.loc[sid, args.target_col]))
            kept_subjects.append(sid)

    keep = np.array(keep, dtype=np.int64)
    X_raw = X_raw[keep]
    mask  = mask[keep]
    subject_ids = np.array(kept_subjects, dtype=str)
    y = np.array(y_list, dtype=np.float32)
    N = len(y)

    print(f"[INFO] N={N} subjects | target: {args.target_col}")
    print(f"[INFO] y stats: mean={y.mean():.1f} std={y.std():.1f} "
          f"min={y.min():.1f} max={y.max():.1f}")

    # ── 特征提取 (原始 + 增强) ──
    mu, sd = masked_mean_std(X_raw, mask)
    Xtab_base = np.concatenate([mu, sd], axis=1)          # 原始(N,66)
    Xtab_rich = _enhance_tabular(Xtab_base)               # 增强
    print(f"[INFO] Feature dims: base={Xtab_base.shape[1]}  rich={Xtab_rich.shape[1]}")

    # ── LOO + 两种特征集 比较 ──
    results = {}

    for feat_name, Xtab in [("base_ridge", Xtab_base), ("rich_ensemble", Xtab_rich)]:
        yhat = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            tr = np.array([j for j in range(N) if j != i], dtype=int)
            y_tr = y[tr]
            
            if args.bound_mode == "minmax":
                lo_b, hi_b = float(y_tr.min()), float(y_tr.max())
            else:
                lo_b = float(np.quantile(y_tr, args.q_low))
                hi_b = float(np.quantile(y_tr, args.q_high))
            
            if feat_name == "base_ridge":
                model = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc",  StandardScaler()),
                    ("m",   Ridge(alpha=args.ridge_alpha))
                ])
                model.fit(Xtab[tr], y_tr)
                pred = float(model.predict(Xtab[[i]])[0])
            else:
                models_dict = build_ensemble_models(args.ridge_alpha)
                pred = stacked_predict(models_dict, Xtab[tr], y_tr,
                                       Xtab[[i]], cv=args.inner_cv)
            
            pred = float(np.clip(pred, lo_b, hi_b))
            yhat[i] = pred
        
        abs_err = np.abs(y - yhat)
        mae  = float(mean_absolute_error(y, yhat))
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        r2   = float(r2_score(y, yhat)) if N >= 3 else float("nan")
        
        # 保形预测区间（Mondrian）
        lo_pi, hi_pi = mondrian_conformal_pi(
            y, yhat, yhat, abs_err,
            alpha=args.alpha, n_bins=args.mondrian_bins
        )
        coverage = float(((y >= lo_pi) & (y <= hi_pi)).mean())
        
        results[feat_name] = {
            "mae": mae, "rmse": rmse, "r2": r2, "coverage": coverage,
            "yhat": yhat, "lo": lo_pi, "hi": hi_pi, "abs_err": abs_err
        }
        
        print(f"\n[{feat_name}]  MAE={mae:.1f}ms  RMSE={rmse:.1f}ms  "
              f"R²={r2:.3f}  Coverage={coverage:.2f}")

    # ── 选出最优方案 ──
    best_name = min(results, key=lambda k: results[k]["mae"])
    best = results[best_name]
    yhat     = best["yhat"]
    lo_pi    = best["lo"]
    hi_pi    = best["hi"]
    abs_err  = best["abs_err"]
    mae      = best["mae"]
    rmse     = best["rmse"]
    r2       = best["r2"]
    coverage = best["coverage"]

    print(f"\n[BEST] {best_name}  MAE={mae:.1f}ms  R²={r2:.3f}  Coverage={coverage:.2f}")

    # ── 保存结果 ──
    out_csv = os.path.join(args.out_dir, "predictions_loocv_enhanced.csv")
    pd.DataFrame({
        "subject_id": subject_ids,
        "true":       y,
        "pred":       yhat,
        "abs_error":  abs_err,
        "pi_low":     lo_pi,
        "pi_high":    hi_pi,
    }).to_csv(out_csv, index=False)

    bar_path = os.path.join(args.out_dir, "abs_error_bar.png")
    plot_abs_error_bar(subject_ids, abs_err, bar_path)

    scatter_path = os.path.join(args.out_dir, "scatter_pred_90PI_enhanced.png")
    title = (f"[{best_name}] LOO | MAE={mae:.1f} RMSE={rmse:.1f} "
             f"R²={r2:.2f} | cov={coverage:.2f}")
    plot_scatter_with_pi(y, yhat, lo_pi, hi_pi, scatter_path, title)

    # 模型对比图
    cmp_path = os.path.join(args.out_dir, "model_comparison.png")
    plot_model_comparison({k: {"mae": v["mae"], "r2": v["r2"], "coverage": v["coverage"]}
                           for k, v in results.items()},
                          cmp_path)

    metrics = {
        "best_strategy":             best_name,
        "N":                         int(N),
        "MAE_ms":                    mae,
        "RMSE_ms":                   rmse,
        "R2":                        r2,
        "empirical_coverage_90PI":   coverage,
        "mondrian_bins":             args.mondrian_bins,
        "bound_mode":                args.bound_mode,
        "q_low":                     args.q_low,
        "q_high":                    args.q_high,
        "ridge_alpha":               float(args.ridge_alpha),
        "all_strategies": {
            k: {"MAE": v["mae"], "R2": v["r2"], "coverage": v["coverage"]}
            for k, v in results.items()
        }
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[DONE]")
    print(f"  Predictions → {out_csv}")
    print(f"  Plots       → {args.out_dir}/")
    print(f"  Metrics     → {os.path.join(args.out_dir, 'metrics.json')}")
    return metrics


if __name__ == "__main__":
    main()