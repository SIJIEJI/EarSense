import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, recall_score
from sklearn.base import clone

from Stress_study_ear_sum.dataInterface.featureDataLoader import FeatureDataLoader
from Stress_study_ear_sum.dataInterface.featureProcessing import FeaturePreprocessor
from Stress_study_ear_sum.dataInterface.dataSplitting import DataSplitting
from Stress_study_ear_sum.machineLearning.classicModels import get_classification_models, get_regression_models

from Stress_study_ear_sum.config import Config


class MLPipeline:
    def __init__(self):
        self.loader = FeatureDataLoader()
        self.preprocessor = FeaturePreprocessor()
        self.splitter = DataSplitting()

    def getSplits(self, yClass, subjects, sessions, splitMethod):
        if splitMethod == 'rolling_origin':
            return list(self.splitter.rollingOriginSplit(sessions, Config.rollingFolds, Config.embargo))
        if splitMethod == 'loso':
            return list(self.splitter.losoSplit(yClass, subjects))
        if splitMethod == 'sgkf':
            return list(self.splitter.stratifiedGroupKFold(yClass, subjects))
        if splitMethod == 'skf':
            return list(self.splitter.stratifiedKFold(yClass))
        raise NotImplementedError(f'Split method {splitMethod!r} not implemented')

    def loadAndProcess(self):
        X, yClass, yReg, subjects, sessions, featureNames = self.loader.loadAllSubjects()
        self.preprocessor.checkData(X)
        X, featureNames = self.preprocessor.dropModalities(X, featureNames)
        print(f"Data loaded: X={X.shape}, subjects={len(np.unique(subjects))}, sessions={len(np.unique(sessions))}, features={len(featureNames)}")
        return X, yClass, yReg, subjects, sessions, featureNames

    def foldMatrices(self, X, yClass, subjects, trainIdx, testIdx):
        Xnorm = self.preprocessor.normalizeBySubjectBaseline(X, yClass, subjects, trainIdx)
        return self.preprocessFold(Xnorm[trainIdx].copy(), Xnorm[testIdx].copy())

    def preprocessFold(self, Xtrain, Xtest):
        usable = self.preprocessor.usableFeatureMask(Xtrain)
        if not usable.all():
            Xtrain[:, ~usable] = 0.0
            Xtest[:, ~usable] = 0.0

        Xtrain = self.preprocessor.imputeNaN(Xtrain, fit=True)
        Xtest = self.preprocessor.imputeNaN(Xtest, fit=False)
        Xtrain = self.preprocessor.clipOutliers(Xtrain, fit=True)
        Xtest = self.preprocessor.clipOutliers(Xtest, fit=False)
        Xtrain = self.preprocessor.standardize(Xtrain, fit=True)
        Xtest = self.preprocessor.standardize(Xtest, fit=False)
        return Xtrain, Xtest

    @staticmethod
    def smoothPredictions(yPred, testIdx, sessions, window):
        if window is None or window <= 1:
            return yPred
        smoothed = yPred.copy()
        testSessions = sessions[testIdx]
        for session in np.unique(testSessions):
            order = np.where(testSessions == session)[0]
            values = yPred[order]
            cumulative = np.concatenate([[0.0], np.cumsum(values)])
            counts = np.arange(1, len(values) + 1)
            low = np.maximum(counts - window, 0)
            smoothed[order] = (cumulative[counts] - cumulative[low]) / (counts - low)
        return smoothed

    def evaluateClassifiers(self, X, yClass, subjects, splits, splitMethod=''):
        models = get_classification_models()
        print(f'\n classification ({splitMethod}): {len(splits)} folds')
        classNames = list(Config.classLabels.keys())
        results = {}
        allPredictions = {}
        allTrainedModels = {}
        for name, model in models.items():
            foldMetrics = []
            yTrueAll, yPredAll = [], []
            trainedModels = []
            for foldId, (trainIdx, testIdx) in enumerate(splits):
                Xtrain, Xtest = self.foldMatrices(X, yClass, subjects, trainIdx, testIdx)
                yClassTrain, yClassTest = yClass[trainIdx], yClass[testIdx]
                modelClone = clone(model)
                modelClone.fit(Xtrain, yClassTrain)
                yPred = modelClone.predict(Xtest)
                yTrueAll.extend(yClassTest)
                yPredAll.extend(yPred)
                foldMetrics.append({
                    'fold': foldId,
                    'test_subject': np.unique(subjects[testIdx]),
                    'n_test': len(testIdx),
                    'accuracy': accuracy_score(yClassTest, yPred),
                    'f1_macro': f1_score(yClassTest, yPred, average='macro'),
                    'f1_weighted': f1_score(yClassTest, yPred, average='weighted')
                })
                trainedModels.append({
                    'model': modelClone,
                    'X_train': Xtrain,
                    'X_test': Xtest,
                    'y_train': yClassTrain,
                    'y_test': yClassTest,
                    'train_idx': trainIdx,
                    'test_idx': testIdx,
                })

            results[name] = foldMetrics
            allPredictions[name] = {'y_true': np.array(yTrueAll), 'y_pred': np.array(yPredAll)}
            allTrainedModels[name] = trainedModels
            meanAcc = np.mean([m['accuracy'] for m in foldMetrics])
            meanF1 = np.mean([m['f1_macro'] for m in foldMetrics])
            perClass = recall_score(yTrueAll, yPredAll, average=None, zero_division=0)
            recallText = '  '.join(f'{c}={r:.2f}' for c, r in zip(classNames, perClass))
            print(f"{name}: acc={meanAcc:.3f}, f1={meanF1:.3f}; recall {recallText}")

        return results, allPredictions, allTrainedModels

    def evaluateRegressors(self, X, yClass, yReg, subjects, sessions, splits, splitMethod='', targetIdx=0):
        # run all regression models for a specific target (0=PA, 1=NA, 2=SA)
        targetName = Config.regTargetNames[targetIdx]
        models = get_regression_models()
        print(f'\n regression on {targetName} ({splitMethod}): {len(splits)} folds')
        y = yReg[:, targetIdx]
        results = {}
        allPredictions = {}
        allTrainedModels = {}
        for name, model in models.items():
            foldMetrics = []
            yTrueAll, yPredAll = [], []
            trainedModels = []
            for foldId, (trainIdx, testIdx) in enumerate(splits):
                Xtrain, Xtest = self.foldMatrices(X, yClass, subjects, trainIdx, testIdx)
                yTrain, yTest = y[trainIdx], y[testIdx]
                # fit on the deviation from each subject's own baseline, score on the raw scale
                offset = (self.preprocessor.baselineTargetOffset(y, yClass, subjects, trainIdx) if Config.centerRegressionTarget else np.zeros(len(y)))
                modelClone = clone(model)
                modelClone.fit(Xtrain, yTrain - offset[trainIdx])
                yPred = self.smoothPredictions(modelClone.predict(Xtest) + offset[testIdx], testIdx, sessions, Config.smoothWindow)
                yTrueAll.extend(yTest)
                yPredAll.extend(yPred)
                foldMetrics.append({
                    'fold': foldId,
                    'test_subject': np.unique(subjects[testIdx]),
                    'n_test': len(testIdx),
                    'mae': mean_absolute_error(yTest, yPred),
                    'r2': r2_score(yTest, yPred),
                })

                trainedModels.append({
                    'model': modelClone,
                    'X_train': Xtrain,
                    'X_test': Xtest,
                    'y_train': yTrain,
                    'y_test': yTest,
                    'train_idx': trainIdx,
                    'test_idx': testIdx,
                })

            results[name] = foldMetrics
            allPredictions[name] = {'y_true': np.array(yTrueAll), 'y_pred': np.array(yPredAll)}
            allTrainedModels[name] = trainedModels

            pooledR2 = r2_score(yTrueAll, yPredAll)
            withinR2 = self.withinConditionR2(np.array(yTrueAll), np.array(yPredAll), yClass, splits)
            print(f"{name}: mae={np.mean([m['mae'] for m in foldMetrics])}, "
                  f"r2={np.mean([m['r2'] for m in foldMetrics])}, "
                  f"pooled r2={pooledR2}, within-condition r2={withinR2}")

        nullR2 = self.conditionMeanNullR2(y, yClass, splits)

        return results, allPredictions, allTrainedModels

    @staticmethod
    def conditionMeanNullR2(y, yClass, splits):
        """R2 of a predictor that is told the true task condition and returns the
        training mean for it. Anything at or below this is explained by condition
        differences alone rather than by continuous affect estimation."""
        yTrueAll, yNullAll = [], []
        for trainIdx, testIdx in splits:
            conditionMeans = {c: y[trainIdx][yClass[trainIdx] == c].mean()
                              for c in np.unique(yClass[trainIdx])}
            fallback = y[trainIdx].mean()
            yTrueAll.extend(y[testIdx])
            yNullAll.extend(conditionMeans.get(c, fallback) for c in yClass[testIdx])
        return r2_score(yTrueAll, yNullAll)

    @staticmethod
    def withinConditionR2(yTrue, yPred, yClass, splits):
        """R2 after each condition's mean is removed from truth and prediction, i.e.
        how much within-condition (between-subject) variance is actually explained."""
        testClasses = np.concatenate([yClass[testIdx] for _, testIdx in splits])
        residualTrue, residualPred = yTrue.astype(float).copy(), yPred.astype(float).copy()
        for c in np.unique(testClasses):
            mask = testClasses == c
            residualTrue[mask] -= yTrue[mask].mean()
            residualPred[mask] -= yPred[mask].mean()
        denominator = np.sum(residualTrue ** 2)
        if denominator == 0:
            return float('nan')
        return 1 - np.sum((residualTrue - residualPred) ** 2) / denominator

    def findBestModel(self, results, metric, exclude=()):
        bestName, bestScore = None, -float('inf')
        for name, foldMetrics in results.items():
            if name in exclude:
                continue
            meanScore = np.mean([m[metric] for m in foldMetrics])
            if meanScore > bestScore:
                bestScore = meanScore
                bestName = name
        return bestName, bestScore