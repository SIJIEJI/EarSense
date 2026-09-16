import os
import numpy as np
import pandas as pd
from Stress_study_ear_sum.config import Config

class FeatureDataLoader:
    def __init__(self, featureDir=None):
        self.config = Config()
        self.featureDir = featureDir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Stress Data', 'SavedFeatures')
        self.metaDataCols = ['subject_id', 'activity', 'condition', 'session', 'session_order', 'cumulative_time', 'timestamp', 'condition_label', 'PA', 'NA', 'SA']

    def loadAllSubjects(self, subjectIds=None):
        if subjectIds is None:
            subjectIds = self.detectSubjects()
        X_all, y_class_all, y_reg_all, subject_all, session_all, featureCols = [], [], [], [], [], []
        sessionCounter = 0
        for subjectId in subjectIds:
            filepath = os.path.join(self.featureDir, f'subject{subjectId}_features.xlsx')
            if not os.path.exists(filepath):
                continue
            df = pd.read_excel(filepath)
            includedSessions = set(self.config.classLabels.keys())
            df = df[df['session'].isin(includedSessions)]
            featureCols = [c for c in df.columns if c not in self.metaDataCols]
            X_all.append(df[featureCols].values)
            y_class_all.append(df['session'].map(self.config.classLabels).values)
            y_reg_all.append(df[['PA', 'NA', 'SA']].values)
            subject_all.append(np.full(len(df), subjectId))
            sessionIds = np.empty(len(df), dtype=int)
            for offset, sessionKey in enumerate(df['session'].unique()):
                sessionIds[(df['session'] == sessionKey).values] = sessionCounter + offset
            sessionCounter += df['session'].nunique()
            session_all.append(sessionIds)

        return (np.vstack(X_all), np.concatenate(y_class_all), np.vstack(y_reg_all),
                np.concatenate(subject_all), np.concatenate(session_all), featureCols)

    def detectSubjects(self):
        subjectIds = []
        for fileName in os.listdir(self.featureDir):
            if fileName.startswith('subject') and fileName.endswith('_features.xlsx'):
                subjectId = int(fileName.replace('subject', '').replace('_features.xlsx', ''))
                subjectIds.append(subjectId)
        return sorted(subjectIds)


if __name__ == '__main__':
    loader = FeatureDataLoader()
    X, y_class, y_reg, subjects, sessions, featureNames = loader.loadAllSubjects()
    print(f'Xshape: {X.shape}, y_class_shape: {y_class.shape}, y_reg_shape: {y_reg.shape}, '
          f'num subjects: {len(np.unique(subjects))}, num sessions: {len(np.unique(sessions))}')