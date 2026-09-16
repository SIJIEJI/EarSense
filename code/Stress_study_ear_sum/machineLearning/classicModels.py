from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


def get_classification_models():
    return {
        "logreg": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        "svm": Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True))]),
        "svm_linear": Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_depth=None, max_features=None, class_weight='balanced_subsample', random_state=42),
        "xgb": XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.7, colsample_bytree=0.7, objective='multi:softprob', eval_metric='mlogloss', random_state=42),
        "lgbm": LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.7, colsample_bytree=0.7, class_weight='balanced', random_state=42, verbose=-1),
        "cat": CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, loss_function='MultiClass', verbose=False, random_seed=42),
        "mlp": Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, max_iter=1000, random_state=42))])
    }


def get_regression_models():
    return {
        "ridge": Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
        "svr": Pipeline([('scaler', StandardScaler()), ('reg', SVR(kernel='rbf', C=10.0, epsilon=0.1))]),
        "svr_linear": Pipeline([('scaler', StandardScaler()), ('reg', SVR(kernel='linear', C=10.0, epsilon=0.1))]),
        "rf_reg": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42),
        "xgb_reg": XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.7, objective='reg:squarederror', random_state=42),
        "lgbm_reg": LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
        "cat_reg": CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, verbose=False, random_seed=42),
    }