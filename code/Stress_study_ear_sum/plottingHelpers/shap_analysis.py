import warnings

import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class ShapAnalyzer:
    def __init__(self, model, model_name, X_train, X_test, feature_names=None, output_dir='./shap_results'):
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model = self._extract_base_model(model)
        self.explainer = None
        self.shap_values = None
        self.X_test_subset = None

    def _extract_base_model(self, model):
        if hasattr(model, 'named_steps'):
            if 'clf' in model.named_steps:
                return model.named_steps['clf']
            elif 'reg' in model.named_steps:
                return model.named_steps['reg']
        return model

    def _is_tree_model(self):
        tree_types = ['RandomForest', 'XGB', 'LGBM', 'CatBoost', 'GradientBoosting']
        model_type = type(self.base_model).__name__
        return any(t in model_type for t in tree_types)

    def _is_linear_model(self):
        linear_types = ['LogisticRegression', 'Ridge', 'Lasso', 'LinearRegression', 'ElasticNet']
        model_type = type(self.base_model).__name__
        return any(t in model_type for t in linear_types)

    def _save_figure(self, filename, fig=None):
        output_path = self.output_dir / f"{filename}.pdf"
        if fig is None:
            fig = plt.gcf()
        fig.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {output_path}")
        plt.close(fig)

    def compute_shap_values(self, max_samples=None):
        self.X_test_subset = self.X_test if max_samples is None else self.X_test[:max_samples]
        if type(self.base_model).__name__ == 'CatBoostClassifier':
            import catboost
            raw = self.base_model.get_feature_importance(catboost.Pool(self.X_test_subset), type='ShapValues')
            self.shap_values = np.transpose(raw[:, :, :-1], (1, 0, 2)) if raw.ndim == 3 else raw[:, :-1]
            return self.shap_values
        if self._is_tree_model():
            self.explainer = shap.TreeExplainer(self.base_model)
        elif self._is_linear_model():
            self.explainer = shap.LinearExplainer(self.base_model, self.X_train)
        else:
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)

        self.shap_values = self.explainer.shap_values(self.X_test_subset)

        shap_array = np.array(self.shap_values)
        if len(shap_array.shape) == 3 and shap_array.shape[0] != shap_array.shape[2]:
            if shap_array.shape[2] < shap_array.shape[1]:
                self.shap_values = np.transpose(shap_array, (2, 0, 1))

        return self.shap_values

    def get_feature_importance_df(self):
        if self.shap_values is None:
            return None
        vals = np.abs(np.array(self.shap_values))
        if len(vals.shape) == 3:
            importance = np.sum(np.mean(vals, axis=1), axis=0)
        else:
            importance = np.mean(vals, axis=0)
        features = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(len(importance))]
        df = pd.DataFrame({'Feature': features, 'Importance': importance})
        return df.sort_values(by='Importance', ascending=False)

    def plot_dependence(self, feature, interaction_idx='auto'):
        if self.shap_values is None:
            return
        shap_array = np.array(self.shap_values)
        is_multiclass = len(shap_array.shape) == 3
        shap_to_plot = shap_array[1] if is_multiclass and shap_array.shape[0] > 1 else (shap_array[0] if is_multiclass else shap_array)
        feat_idx = self.feature_names.index(feature) if self.feature_names and feature in self.feature_names else int(feature)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(feat_idx, shap_to_plot, self.X_test_subset, feature_names=self.feature_names, interaction_index=interaction_idx, show=False, ax=ax, alpha=0.6, dot_size=20)
        ax.set_title(f"{feature}", fontsize=13, fontweight='bold', pad=10)
        self._save_figure(f"{self.model_name}_dependence_{feature}", fig)

    def plot_decision(self, num_samples=1000):
        if self.shap_values is None or self.explainer is None:
            print("SHAP values or explainer not available for decision plot.")
            return
        shap_array = np.array(self.shap_values)
        is_multiclass = len(shap_array.shape) == 3
        if is_multiclass:
            print("Decision plot skipped for multiclass classification.")
            return
        if num_samples is not None and len(self.X_test_subset) > num_samples:
            sample_idx = np.random.choice(len(self.X_test_subset), num_samples, replace=False)
            shap_subset = shap_array[sample_idx]
            X_subset = self.X_test_subset[sample_idx]
        else:
            shap_subset = shap_array
            X_subset = self.X_test_subset

        n_samples = len(X_subset)
        alpha = 0.1 if n_samples > 1000 else (0.2 if n_samples > 500 else 0.3)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.decision_plot(self.explainer.expected_value, shap_subset, X_subset, feature_names=self.feature_names, show=False, alpha=alpha, highlight=None, ignore_warnings=True)
        ax.set_title(f"SHAP Decision Plot (n={n_samples})", fontsize=13, fontweight='bold', pad=10)
        self._save_figure(f"{self.model_name}_decision_plot", fig)

    def plot_summary(self, class_names=None, max_display=20):
        if self.shap_values is None:
            return
        shap_array = np.array(self.shap_values)
        is_multiclass = len(shap_array.shape) == 3
        if is_multiclass and class_names:
            shap_list = [shap_array[i] for i in range(shap_array.shape[0])]
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_list, self.X_test_subset, feature_names=self.feature_names, class_names=class_names, max_display=max_display, show=False)
            self._save_figure(f"{self.model_name}_shap_summary_multiclass", fig)
        else:
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(self.shap_values, self.X_test_subset, feature_names=self.feature_names, max_display=max_display, show=False)
            plt.title("SHAP Feature Impact", fontsize=13, fontweight='bold', pad=10)
            self._save_figure(f"{self.model_name}_shap_summary", fig)

    def create_publication_figure(self, class_names=None, max_display=20):
        if self.shap_values is None:
            return
        shap_array = np.array(self.shap_values)
        is_multiclass = len(shap_array.shape) == 3
        importance_df = self.get_feature_importance_df()
        top_features = importance_df['Feature'].head(2).tolist()
        if is_multiclass:
            target_class_idx = 1 if shap_array.shape[0] > 1 else 0
            shap_to_plot = shap_array[target_class_idx]
        else:
            shap_to_plot = shap_array

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.sca(ax1)
        shap.summary_plot(shap_to_plot, self.X_test_subset, feature_names=self.feature_names, max_display=max_display, show=False, plot_size=None)
        ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax1.set_title('Feature Impact', fontsize=11, fontweight='bold')

        ax2 = fig.add_subplot(gs[0, 1])
        plt.sca(ax2)
        shap.summary_plot(shap_to_plot, self.X_test_subset, feature_names=self.feature_names, plot_type="bar", max_display=max_display, show=False, plot_size=None, color='#3C5488')
        ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.set_title('Global Importance', fontsize=11, fontweight='bold')

        if top_features:
            ax3 = fig.add_subplot(gs[1, 0])
            plt.sca(ax3)
            feat_idx = self.feature_names.index(top_features[0])
            shap.dependence_plot(feat_idx, shap_to_plot, self.X_test_subset, feature_names=self.feature_names, interaction_index='auto', show=False, alpha=0.6, dot_size=15, ax=ax3)
            ax3.set_title(f'{top_features[0]}', fontsize=11, fontweight='bold')
            ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')

        if len(top_features) > 1:
            ax4 = fig.add_subplot(gs[1, 1])
            plt.sca(ax4)
            feat_idx = self.feature_names.index(top_features[1])
            shap.dependence_plot(feat_idx, shap_to_plot, self.X_test_subset, feature_names=self.feature_names, interaction_index='auto', show=False, alpha=0.6, dot_size=15, ax=ax4)
            ax4.set_title(f'{top_features[1]}', fontsize=11, fontweight='bold')
            ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top')
        self._save_figure(f"{self.model_name}_comprehensive_summary", fig)


def analyze_best_model(model, model_name, X_train, X_test, y_test=None, feature_names=None, class_names=None, output_dir='./shap_results'):
    analyzer = ShapAnalyzer(model, model_name, X_train, X_test, feature_names, output_dir)
    analyzer.compute_shap_values()
    analyzer.plot_summary(class_names=class_names)
    analyzer.create_publication_figure(class_names=class_names)
    return analyzer


def analyze_best_model_all_folds(fold_data_list, model_name, feature_names=None, class_names=None, output_dir='./shap_results'):
    all_shap_values = []
    all_test_data = []
    all_explainers = []

    for i, fold_data in enumerate(fold_data_list):
        analyzer = ShapAnalyzer(model=fold_data['model'], model_name=f"{model_name}_fold_{i}", X_train=fold_data['X_train'], X_test=fold_data['X_test'], feature_names=feature_names, output_dir=output_dir)
        shap_vals = analyzer.compute_shap_values()
        if shap_vals is not None:
            all_shap_values.append(shap_vals)
            all_test_data.append(fold_data['X_test'])
            all_explainers.append(analyzer.explainer)
    first_shap = np.array(all_shap_values[0])
    is_multiclass = len(first_shap.shape) == 3

    if is_multiclass:
        concatenated_shap = np.concatenate(all_shap_values, axis=1)
    else:
        concatenated_shap = np.concatenate(all_shap_values, axis=0)
    X_test_concatenated = np.concatenate(all_test_data, axis=0)
    agg_output_dir = Path(output_dir) / "aggregated"
    agg_analyzer = ShapAnalyzer(model=fold_data_list[0]['model'], model_name=f"{model_name}_Aggregated", X_train=fold_data_list[0]['X_train'], X_test=X_test_concatenated, feature_names=feature_names, output_dir=agg_output_dir)
    agg_analyzer.shap_values = concatenated_shap
    agg_analyzer.X_test_subset = X_test_concatenated
    agg_analyzer.explainer = all_explainers[0]

    agg_analyzer.plot_summary(class_names=class_names, max_display=20)
    agg_analyzer.create_publication_figure(class_names=class_names, max_display=20)

    if not is_multiclass:
        agg_analyzer.plot_decision(num_samples=None)
    importance_df = agg_analyzer.get_feature_importance_df()
    if importance_df is not None:
        top_features = importance_df['Feature'].head(5).tolist()
        for feat in top_features:
            agg_analyzer.plot_dependence(feat, interaction_idx='auto')

    return {
        'concatenated_shap_values': concatenated_shap,
        'concatenated_test_data': X_test_concatenated
    }
