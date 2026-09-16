import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, StratifiedGroupKFold


class DataSplitting:
    @staticmethod
    def _placeholder(n):
        return np.zeros((n, 1))

    def losoSplit(self, yClass, subjects):
        for trainIdx, testIdx in LeaveOneGroupOut().split(self._placeholder(len(yClass)), yClass, subjects):
            yield trainIdx, testIdx

    def stratifiedKFold(self, yClass, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for trainIdx, testIdx in skf.split(self._placeholder(len(yClass)), yClass):
            yield trainIdx, testIdx

    def stratifiedGroupKFold(self, yClass, subjects, n_splits=5):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for trainIdx, testIdx in sgkf.split(self._placeholder(len(yClass)), yClass, subjects):
            yield trainIdx, testIdx

    def rollingOriginSplit(self, sessions, n_splits=4, embargo=1):
        if n_splits < 1:
            raise ValueError('rollingOriginSplit requires n_splits >= 1')
        if embargo < 0:
            raise ValueError('rollingOriginSplit requires separation >= 0')
        edges = np.linspace(0.0, 1.0, n_splits + 2)[1:]
        for k in range(n_splits):
            trainIdx, testIdx = [], []
            for session in np.unique(sessions):
                # ascending index order is acquisition order within a session
                sessionIdx = np.where(sessions == session)[0]
                start = int(round(len(sessionIdx) * edges[k]))
                stop = int(round(len(sessionIdx) * edges[k + 1]))
                trainStop = start - embargo
                if stop <= start or trainStop <= 0:
                    continue
                trainIdx.extend(sessionIdx[:trainStop])
                testIdx.extend(sessionIdx[start:stop])
            if not testIdx:
                raise ValueError('rollingOriginSplit produced an empty test fold, reduce n_splits number')
            yield np.array(sorted(trainIdx)), np.array(sorted(testIdx))