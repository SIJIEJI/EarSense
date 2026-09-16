import pandas as pd
import numpy as np
from pathlib import Path

from Stress_study_ear_sum.config import Config


def cleanSignal(timeColumn, valueColumns, gapMultiplier=5.0):
    timeArray = np.asarray(timeColumn, dtype=float)
    values = np.asarray(valueColumns, dtype=float)
    if values.ndim == 1:
        valid = np.isfinite(timeArray) & np.isfinite(values)
    else:
        valid = np.isfinite(timeArray) & np.all(np.isfinite(values), axis=1)
    timeArray = timeArray[valid]
    values = values[valid]
    if timeArray.size < 2:
        return None, None, None, None
    order = np.argsort(timeArray)
    timeArray = timeArray[order]
    values = values[order]
    dt = np.diff(timeArray)
    positiveTimeDiff = dt[dt > 0]
    if positiveTimeDiff.size == 0:
        return None, None, None, None
    medianTimeDiff = np.median(positiveTimeDiff)
    gapThreshold = gapMultiplier * medianTimeDiff
    gapMask = dt > gapThreshold
    gapIndices = np.where(gapMask)[0]
    normalTimeDiff = positiveTimeDiff[~gapMask[dt > 0]]
    if normalTimeDiff.size == 0:
        normalTimeDiff = positiveTimeDiff
    fs = 1.0 / np.median(normalTimeDiff)
    return timeArray, values, fs, gapIndices


class StressDataLoader:
    def __init__(self, dataRoot):
        self.dataRoot = Path(dataRoot)
        self.stressDataPath = self.dataRoot
        self.activities = ['CPT', 'VR']
        self.conditions = ['Baseline', 'Stressor', 'Recovery']
        self.subjects = self.getSubjects()

        self.activityOnehot = {}
        self.conditionOnehot = {}
        self.numActivities = len(self.activities)
        self.numConditions = len(self.conditions)
        self.combinedLabels = []
        self.combinedOnehot = {}
        self.mapActivityToID()

    def getSubjects(self):
        subjects = []
        for folder in self.stressDataPath.iterdir():
            if folder.is_dir() and folder.name.startswith('Subject_'):
                subjects.append(int(folder.name.split('_')[1]))
        return sorted(subjects)

    def mapActivityToID(self):
        for idx, activityType in enumerate(self.activities):
            onehot = np.zeros(self.numActivities)
            onehot[idx] = 1
            self.activityOnehot[activityType] = onehot.tolist()
        for idx, conditionType in enumerate(self.conditions):
            onehot = np.zeros(self.numConditions)
            onehot[idx] = 1
            self.conditionOnehot[conditionType] = onehot.tolist()
        for activityType in self.activities:
            for conditionType in self.conditions:
                self.combinedLabels.append(f"{activityType}_{conditionType}")
        for idx, labelType in enumerate(self.combinedLabels):
            numCombined = len(self.combinedLabels)
            onehot = np.zeros(numCombined)
            onehot[idx] = 1
            self.combinedOnehot[labelType] = onehot.tolist()

    def loadQuestionnaire(self, subjectId, activityName, conditionName):
        subjectPath = self.stressDataPath / f"Subject_{subjectId}"
        filename = f"Q_{activityName}_{conditionName}.xlsx"
        filepath = subjectPath / filename
        if not filepath.exists():
            return None
        panas = pd.read_excel(filepath, sheet_name='PANAS')
        stai = pd.read_excel(filepath, sheet_name='STAI-Y1')
        PA, NA, SA = StressDataLoader.PanasSTAICalculation(panas.values.tolist(), stai.values.tolist())
        return {
            'PANAS': panas.values.tolist(),
            'STAI-Y1': stai.values.tolist(),
            'PA': PA,
            'NA': NA,
            'SA': SA,
            'activityOnehot': self.activityOnehot[activityName],
            'conditionOnehot': self.conditionOnehot[conditionName],
            'combinedOnehot': self.combinedOnehot[f'{activityName}_{conditionName}'],
        }

    @staticmethod
    def PanasSTAICalculation(panasList, staiList):
        PA_score = 0
        NA_score = 0
        SA_score_half = 0
        SA_reversed_score = 0
        panas_scores = [row[1] for row in panasList]
        stai_scores = [row[1] for row in staiList]
        for i in [0, 2, 4, 8, 9, 11, 13, 15, 16, 18]:
            PA_score += int(panas_scores[i])
        for j in [1, 3, 5, 6, 7, 10, 12, 14, 17, 19]:
            NA_score += int(panas_scores[j])
        for k in [0, 1, 4, 7, 9, 10, 14, 15, 18, 19]:
            stai_val = int(stai_scores[k])
            if stai_val == 4:
                SA_reversed_score += 1
            elif stai_val == 3:
                SA_reversed_score += 2
            elif stai_val == 2:
                SA_reversed_score += 3
            elif stai_val == 1:
                SA_reversed_score += 4
        for l in [2, 3, 5, 6, 8, 11, 12, 13, 16, 17]:
            SA_score_half += int(stai_scores[l])
        SA_score = SA_score_half + SA_reversed_score
        assert 10 <= PA_score <= 50 and 10 <= NA_score <= 50
        assert 20 <= SA_score <= 80
        return PA_score, NA_score, SA_score

    def loadEEGECG(self, subjectId, activityName, conditionName):
        subject_path = self.stressDataPath / f"Subject_{subjectId}"
        filename = f"EEG_{activityName}_{conditionName}.txt"
        filepath = subject_path / filename
        if not filepath.exists():
            return None
        data = pd.read_csv(filepath, sep='\t', header=None)
        rawTime = data.iloc[:, 0].values
        ecg = data.iloc[:, 1].values
        eeg = data.iloc[:, 2].values

        ecgTime, ecgClean, ecgFs, ecgGaps = cleanSignal(rawTime, ecg)
        eegTime, eegClean, eegFs, eegGaps = cleanSignal(rawTime, eeg)
        ecgNormalizedTime = ecgTime - ecgTime[0] # ecg and eeg share the time axis
        if len(ecgGaps) > 0 or len(eegGaps) > 0:
            print(f'eeg/ecg gap detected for {filename}')

        if ecgTime is None or eegTime is None:
            return None
        return {
            'time': ecgNormalizedTime.tolist(),
            'ECG': ecgClean.tolist(),
            'EEG': eegClean.tolist(),
            'samplingRate': float(ecgFs),
            'gapIndicesECG': ecgGaps.tolist(),
            'gapIndicesEEG': eegGaps.tolist(),
            'activityOnehot': self.activityOnehot[activityName],
            'conditionOnehot': self.conditionOnehot[conditionName],
            'combinedOnehot': self.combinedOnehot[f'{activityName}_{conditionName}'],
        }

    def loadPeripheral(self, subjectID, activityName, conditionName):
        subjectPath = self.stressDataPath / f"Subject_{subjectID}"
        filename = f"PPG_{activityName}_{conditionName}.xlsx"
        filepath = subjectPath / filename
        if not filepath.exists():
            return None
        df = pd.read_excel(filepath)
        result = {}
        ppg_df = df.iloc[:, 5:9].dropna(subset=df.columns[5:9], how='all')
        if len(ppg_df) > 0:
            ppgValues = ppg_df.iloc[:, 1:4].values
            validMask = np.all(np.isfinite(ppgValues), axis=1)
            ppgValid = ppgValues[validMask]
            if len(ppgValid) > 0:
                fsPPG = 100.0
                numSamples = len(ppgValid)
                ppgTime = np.arange(numSamples) / fsPPG
                result['PPG'] = {
                    'time': ppgTime.tolist(),
                    'dataCh1': ppgValid[:, 0].tolist(),
                    'dataCh2': ppgValid[:, 1].tolist(),
                    'dataCh3': ppgValid[:, 2].tolist(),
                    'samplingFreq': float(fsPPG),
                    'gapIndices': [],
                    'activityOnehot': self.activityOnehot[activityName],
                    'conditionOnehot': self.conditionOnehot[conditionName],
                    'combinedOnehot': self.combinedOnehot[f'{activityName}_{conditionName}'],
                }

        gsr_df = df.iloc[:, 9:11].dropna(subset=df.columns[9:11], how='all')
        if len(gsr_df) > 0:
            gsrTime, gsrValue, fsGSR, gaps = cleanSignal(gsr_df.iloc[:, 0].values, gsr_df.iloc[:, 1].values)
            gsrNormalizedTime = gsrTime - gsrTime[0]
            if len(gaps) > 0:
                print(f'gsr gap detected for {filename}')

            if gsrTime is not None:
                result['GSR'] = {
                    'time': gsrNormalizedTime.tolist(),
                    'data': gsrValue.tolist(),
                    'samplingFreq': float(fsGSR),
                    'gapIndices': gaps.tolist(),
                    'activityOnehot': self.activityOnehot[activityName],
                    'conditionOnehot': self.conditionOnehot[conditionName],
                    'combinedOnehot': self.combinedOnehot[f'{activityName}_{conditionName}'],
                }

        temp_df = df.iloc[:, 11:13].dropna(subset=df.columns[11:13], how='all')
        if len(temp_df) > 0:
            tempTime, tempValue, fsTemp, gaps = cleanSignal(temp_df.iloc[:, 0].values, temp_df.iloc[:, 1].values)
            tempNormalizedTime = tempTime - tempTime[0]
            if len(gaps) > 0:
                print(f'temp gap detected for {filename}')

            if tempTime is not None:
                result['Temp'] = {
                    'time': tempNormalizedTime.tolist(),
                    'data': tempValue.tolist(),
                    'samplingFreq': float(fsTemp),
                    'gapIndices': gaps.tolist(),
                    'activityOnehot': self.activityOnehot[activityName],
                    'conditionOnehot': self.conditionOnehot[conditionName],
                    'combinedOnehot': self.combinedOnehot[f'{activityName}_{conditionName}'],
                }

        acc_df = df.iloc[:, 13:17].dropna(subset=df.columns[13:17], how='all')
        if len(acc_df) > 0:
            accTime, accValue, fsACC, gaps = cleanSignal(acc_df.iloc[:, 0].values, acc_df.iloc[:, 1:4].values)
            accNormalizedTime = accTime - accTime[0]
            if len(gaps) > 0:
                print(f'acc gap detected for {filename}')

            if accTime is not None:
                result['ACC'] = {
                    'time': accNormalizedTime.tolist(),
                    'x': accValue[:, 0].tolist(),
                    'y': accValue[:, 1].tolist(),
                    'z': accValue[:, 2].tolist(),
                    'samplingFreq': float(fsACC),
                    'gapIndices': gaps.tolist(),
                    'activityOnehot': self.activityOnehot[activityName],
                    'conditionOnehot': self.conditionOnehot[conditionName],
                    'combinedOnehot': self.combinedOnehot[f'{activityName}_{conditionName}'],
                }
        return result

    def loadSingleSubject(self, subjectID, modalities=None):
        if modalities is None:
            modalities = Config.fileModalities
        data = {}
        for modality in modalities:
            data[modality] = {}
            for activityName in self.activities:
                for conditionName in self.conditions:
                    key = f"{activityName}_{conditionName}"
                    if modality == 'questionnaire':
                        result = self.loadQuestionnaire(subjectID, activityName, conditionName)
                    elif modality == 'eeg_ecg':
                        result = self.loadEEGECG(subjectID, activityName, conditionName)
                    elif modality == 'peripheral':
                        result = self.loadPeripheral(subjectID, activityName, conditionName)
                    else:
                        result = None
                    if result is not None:
                        data[modality][key] = result
                    else:
                        raise ValueError('Missing modality data, double check files')
        return data

    def loadAllSubjects(self, subjects=None, modalities=None):
        if subjects is None:
            subjects = self.subjects
        allData = {}
        for subjectID in subjects:
            print(f"Loading Subject {subjectID}...")
            allData[subjectID] = self.loadSingleSubject(subjectID, modalities)
        return allData


if __name__ == "__main__":
    data_root = Path(__file__).parent.parent / 'Stress Data'
    loader = StressDataLoader(data_root)
    print(f"Available subjects: {loader.subjects}")