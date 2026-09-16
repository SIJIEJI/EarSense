import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
from scipy import stats

TARGET_SCORE_RANGES = {
    'PA': (10, 41),
    'NA': (8, 40),
    'SA': (20, 70),
    'STAI': (20, 70),
}
TARGET_TICK_STEP = 10


def get_target_score_range(target_name):
    return TARGET_SCORE_RANGES.get(str(target_name).strip().upper())


def _model_tick_labels(model_names):
    return [name.replace('_', '\n') for name in model_names]


def _apply_score_ticks(ax, min_val, max_val, tick_step):
    if not tick_step:
        return
    major = float(tick_step)
    while (max_val - min_val) / major > 16:
        major *= 2
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(major))
        if major != tick_step:
            axis.set_minor_locator(MultipleLocator(tick_step))


def set_publication_style():
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })


def plot_confusion_matrices(y_true_dict, y_pred_dict, class_names, save_path='confusion_matrices.pdf'):
    set_publication_style()
    n_models = len(y_true_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 3.5))
    if n_models == 1:
        axes = [axes]
    cmap = plt.get_cmap('Blues')
    norm = Normalize(vmin=0, vmax=1)
    n_classes = len(class_names)

    for idx, (model_name, y_true) in enumerate(y_true_dict.items()):
        y_pred = y_pred_dict[model_name]
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax = axes[idx]
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(cm_normalized[i, j])), edgecolor='white', linewidth=0.5))
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})', ha="center", va="center", color="white" if cm_normalized[i, j] > thresh else "black", fontsize=9)
        ax.set_xlim(-0.5, n_classes - 0.5)
        ax.set_ylim(n_classes - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'{model_name.upper()}', fontweight='bold')
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            fraction=0.046, pad=0.04)
        if cbar.solids is not None:
            cbar.solids.set_rasterized(False)
        cbar.set_label('Normalized Accuracy', rotation=270, labelpad=20)

    plt.suptitle('Classification Confusion Matrices', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved confusion matrices to {save_path}")
    plt.close()


def plot_f1_comparison(classification_results, save_path='f1_comparison.pdf'):
    set_publication_style()
    model_names = []
    f1_means = []
    f1_stds = []

    for model_name, fold_metrics in classification_results.items():
        f1_scores = [m['f1_macro'] for m in fold_metrics]
        model_names.append(model_name.upper())
        f1_means.append(np.mean(f1_scores))
        f1_stds.append(np.std(f1_scores))
    sorted_indices = np.argsort(f1_means)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    f1_means = [f1_means[i] for i in sorted_indices]
    f1_stds = [f1_stds[i] for i in sorted_indices]
    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.arange(len(model_names))
    colors = ['#A9A9A9'] * len(model_names)
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.0)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1-Macro Score', fontsize=12)
    ax.set_title('Classification Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(_model_tick_labels(model_names), rotation=0)
    ax.set_ylim(0, min(1.1, max(f1_means) + max(f1_stds) + 0.1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved F1 comparison to {save_path}")
    plt.close()


def plot_regression_results(y_true_dict, y_pred_dict, save_path='regression_results.pdf', target_name='STAI', score_range=None, tick_step=TARGET_TICK_STEP):
    set_publication_style()
    n_models = len(y_true_dict)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    if score_range is None:
        score_range = get_target_score_range(target_name)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8*n_cols, 5.2*n_rows), squeeze=False)
    axes = axes.flatten()
    for idx, (model_name, y_true) in enumerate(y_true_dict.items()):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred_dict[model_name], dtype=float).ravel()
        ax = axes[idx]
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        line_x = np.linspace(y_true.min(), y_true.max(), 100)
        line_y = slope * line_x + intercept
        n = len(y_true)
        residuals = y_pred - (slope * y_true + intercept)
        residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))
        y_mean = np.mean(y_true)
        ssx = np.sum((y_true - y_mean)**2)
        prediction_se = residual_std * np.sqrt(1 + 1/n + (line_x - y_mean)**2 / ssx)
        from scipy.stats import t as t_dist
        t_val = t_dist.ppf(0.975, n - 2)  # 95% CI
        margin = t_val * prediction_se
        band_low, band_high = line_y - margin, line_y + margin
        if score_range is None:
            lo = float(min(y_true.min(), y_pred.min(), band_low.min()))
            hi = float(max(y_true.max(), y_pred.max(), band_high.max()))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            lo, hi = lo - pad, hi + pad
            if tick_step:
                lo = np.floor(lo / tick_step) * tick_step
                hi = np.ceil(hi / tick_step) * tick_step
            min_val, max_val = float(lo), float(hi)
        else:
            min_val, max_val = float(score_range[0]), float(score_range[1])
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.plot(line_x, line_y, 'b-', linewidth=2, alpha=0.7, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
        ax.fill_between(line_x, band_low, band_high, color='blue', alpha=0.15, label='95% Prediction Interval')
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_xlabel(f'True {target_name} Score', fontweight='bold')
        ax.set_ylabel(f'Predicted {target_name} Score', fontweight='bold')
        ax.set_title(f'{model_name.upper()}\nR²={r2:.3f}, MAE={mae:.2f}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')
        _apply_score_ticks(ax, min_val, max_val, tick_step)
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    plt.suptitle(f'Regression Performance: Predicted vs Actual {target_name} Scores', fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved regression results to {save_path}")
    plt.close()


def plot_regression_metrics_comparison(regression_results, save_path='regression_comparison.pdf', target_name='STAI'):
    set_publication_style()
    model_names = []
    r2_means = []
    r2_stds = []
    mae_means = []
    mae_stds = []

    for model_name, fold_metrics in regression_results.items():
        r2_scores = [m['r2'] for m in fold_metrics]
        mae_scores = [m['mae'] for m in fold_metrics]
        model_names.append(model_name.upper())
        r2_means.append(np.mean(r2_scores))
        r2_stds.append(np.std(r2_scores))
        mae_means.append(np.mean(mae_scores))
        mae_stds.append(np.std(mae_scores))
    sorted_indices = np.argsort(r2_means)[::-1]
    model_names_sorted = [model_names[i] for i in sorted_indices]
    r2_means_sorted = [r2_means[i] for i in sorted_indices]
    r2_stds_sorted = [r2_stds[i] for i in sorted_indices]
    mae_means_sorted = [mae_means[i] for i in sorted_indices]
    mae_stds_sorted = [mae_stds[i] for i in sorted_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    x = np.arange(len(model_names_sorted))
    colors = ['#A9A9A9'] * len(model_names)
    bars1 = ax1.bar(x, r2_means_sorted, yerr=r2_stds_sorted, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.0)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('(A) Coefficient of Determination (R2)',fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(_model_tick_labels(model_names_sorted), rotation=0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    bars2 = ax2.bar(x, mae_means_sorted, yerr=mae_stds_sorted, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.0)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel(f'Mean Absolute Error ({target_name} points)', fontsize=12)
    ax2.set_title('(B) Mean Absolute Error (MAE)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(_model_tick_labels(model_names_sorted), rotation=0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    plt.suptitle('Regression Performance Metrics Comparison', fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved regression metrics comparison to {save_path}")
    plt.close()


def create_combined_report(classification_results, regression_results, cls_y_true_dict, cls_y_pred_dict, reg_y_true_dict, reg_y_pred_dict, class_names=['CPT_Baseline', 'CPT_Stressor', 'VR_Stressor', 'VR_Recovery'], output_dir='./', target_name='STAI'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrices(cls_y_true_dict, cls_y_pred_dict, class_names, save_path=os.path.join(output_dir, 'confusion_matrices.pdf'))
    plot_f1_comparison(classification_results, save_path=os.path.join(output_dir, 'f1_comparison.pdf'))
    plot_regression_results(reg_y_true_dict, reg_y_pred_dict, save_path=os.path.join(output_dir, 'regression_scatter.pdf'), target_name=target_name)
    plot_regression_metrics_comparison(regression_results, save_path=os.path.join(output_dir, 'regression_metrics.pdf'), target_name=target_name)
    print(f"\nOutput directory: {output_dir}")
