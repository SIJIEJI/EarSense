import numpy as np
from sklearn.preprocessing import StandardScaler
from Stress_study_ear_sum.config import Config


class FeaturePreprocessor:
    def __init__(self):
        self.scaler = None
        self.fillValues = None
        self.outlierBounds = None

    def checkData(self, X):
        print(f"Shape: {X.shape}")
        print(f"NaN count: {np.isnan(X).sum()}")
        print(f"Inf count: {np.isinf(X).sum()}")
        print(f"NaN per feature: {np.isnan(X).any(axis=0).sum()} features affected")

    # Before data splitting
    def dropModalities(self, X, featureNames, excludeModalities=None):
        excludeModalities = Config.excludeModalities if excludeModalities is None else excludeModalities
        if not excludeModalities:
            return X, featureNames
        prefixes = {key.lower(): value for key, value in Config.modalityPrefixes.items()}
        unknown = [m for m in excludeModalities if m.lower() not in prefixes]
        if unknown:
            raise ValueError(f'Unknown modalities {unknown}, expected {list(Config.modalityPrefixes)}')
        dropPrefixes = tuple(prefixes[m.lower()] for m in excludeModalities)
        keep = [not name.startswith(dropPrefixes) for name in featureNames]
        print(f'Dropped {len(keep) - sum(keep)} features from modalities {list(excludeModalities)}')
        return X[:, keep], [name for name, k in zip(featureNames, keep) if k]

    def removeConstantFeatures(self, X, featureNames):
        variances = np.nanvar(X, axis=0)
        featureMask = variances > 1e-10
        print(f'Removed {(~featureMask).sum()} constant features')
        return X[:, featureMask], [name for name, keep in zip(featureNames, featureMask) if keep]

    def normalizeBySubjectBaseline(self, X, yClass, subjects, trainIdx=None):
        X = X.copy()
        reference = np.ones(len(X), dtype=bool)
        if trainIdx is not None:
            reference = np.zeros(len(X), dtype=bool)
            reference[trainIdx] = True

        for subject in np.unique(subjects):
            subjectMask = subjects == subject
            baselineMask = subjectMask & reference & np.isin(yClass, Config.baselineClasses)
            if baselineMask.sum() == 0:
                baselineMask = subjectMask & np.isin(yClass, Config.baselineClasses)
            if baselineMask.sum() > 0:
                muBaseline = X[baselineMask].mean(axis=0)
                sigmaBaseline = X[baselineMask].std(axis=0)
                sigmaBaseline[sigmaBaseline < 1e-10] = 1
                X[subjectMask] = (X[subjectMask] - muBaseline) / sigmaBaseline
        return X

    @staticmethod
    def usableFeatureMask(Xtrain):
        with np.errstate(invalid='ignore'):
            variances = np.nanvar(Xtrain, axis=0)
        return np.isfinite(variances) & (variances > 1e-10)

    @staticmethod
    def baselineTargetOffset(y, yClass, subjects, trainIdx):
        offset = np.zeros(len(y), dtype=float)
        inTrain = np.zeros(len(y), dtype=bool)
        inTrain[trainIdx] = True
        globalMean = y[inTrain].mean()

        for subject in np.unique(subjects):
            subjectMask = subjects == subject
            baselineMask = subjectMask & inTrain & np.isin(yClass, Config.baselineClasses)
            if baselineMask.sum() > 0:
                offset[subjectMask] = y[baselineMask].mean()
            elif (subjectMask & inTrain).sum() > 0:
                offset[subjectMask] = y[subjectMask & inTrain].mean()
            else:
                offset[subjectMask] = globalMean
        return offset

    # after split
    def imputeNaN(self, X, fit=True):
        X = X.copy()
        X[np.isinf(X)] = np.nan
        if fit:
            self.fillValues = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = self.fillValues[i]
        return X

    def clipOutliers(self, X, fit=True, threshold=5):
        if fit:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std < 1e-10] = 1
            self.outlierBounds = (mean - threshold * std, mean + threshold * std)
        return np.clip(X, self.outlierBounds[0], self.outlierBounds[1])

    def standardize(self, X, fit=True):
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
