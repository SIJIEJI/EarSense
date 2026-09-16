import numpy as np
import pandas as pd

from Stress_study_ear_sum.BioSignalAnalysis.AccAnalysis import AccAnalysis
from Stress_study_ear_sum.BioSignalAnalysis.ECGAnalysis import ECGAnalysis
from Stress_study_ear_sum.BioSignalAnalysis.EEGAnalysis import EEGAnalysis
from Stress_study_ear_sum.BioSignalAnalysis.PPGAnalysis import PPGAnalysis
from Stress_study_ear_sum.BioSignalAnalysis.GSRAnalysis import GSRAnalysis
from Stress_study_ear_sum.BioSignalAnalysis.TempAnalysis import TempAnalysis
from Stress_study_ear_sum.config import Config


class FeatureOrganizer:
    sessionOrder = ['CPT_Baseline', 'CPT_Stressor', 'CPT_Recovery', 'VR_Baseline', 'VR_Stressor', 'VR_Recovery']
    featureNames = {
        'ACC': ['acc_mean_x', 'acc_mean_y', 'acc_mean_z', 'acc_mean_mag',
                'acc_std_x', 'acc_std_y', 'acc_std_z', 'acc_std_mag',
                'acc_absint_x', 'acc_absint_y', 'acc_absint_z', 'acc_absint_mag',
                'acc_peakfreq_x', 'acc_peakfreq_y', 'acc_peakfreq_z'],
        'ECG': ['ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_mean', 'ecg_rr_std',
                'ecg_rmssd', 'ecg_nn50', 'ecg_pnn50',
                'ecg_lf_power', 'ecg_hf_power', 'ecg_uhf_power', 'ecg_total_power',
                'ecg_lf_hf_ratio', 'ecg_rel_lf', 'ecg_rel_hf', 'ecg_rel_uhf'],
        'EEG': ['eeg_hjorth_activity', 'eeg_hjorth_mobility', 'eeg_hjorth_complexity',
                'eeg_delta', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma', 'eeg_total_power',
                'eeg_rel_delta', 'eeg_rel_theta', 'eeg_rel_alpha', 'eeg_rel_beta', 'eeg_rel_gamma',
                'eeg_alpha_beta', 'eeg_theta_alpha', 'eeg_theta_beta', 'eeg_alpha_theta',
                'eeg_beta_alpha', 'eeg_delta_beta', 'eeg_delta_alpha', 'eeg_engagement',
                'eeg_spectral_entropy', 'eeg_mean_freq', 'eeg_peak_freq',
                'eeg_mean', 'eeg_std', 'eeg_var', 'eeg_min', 'eeg_max', 'eeg_range',
                'eeg_rms', 'eeg_skew', 'eeg_kurt', 'eeg_zcr'],
        'PPG': ['ppg_pr_mean', 'ppg_pr_std', 'ppg_pr_min', 'ppg_pr_max', 'ppg_pr_range',
                'ppg_ibi_mean', 'ppg_ibi_std', 'ppg_rmssd', 'ppg_pnn50',
                'ppg_rr_mean', 'ppg_rr_std', 'ppg_rr_amp_mean', 'ppg_rr_amp_std',
                'ppg_green_mean', 'ppg_green_std', 'ppg_green_min', 'ppg_green_max', 'ppg_green_range',
                'ppg_red_mean', 'ppg_red_std', 'ppg_red_min', 'ppg_red_max', 'ppg_red_range', 'ppg_red_absint',
                'ppg_ir_mean', 'ppg_ir_std', 'ppg_ir_min', 'ppg_ir_max', 'ppg_ir_range', 'ppg_ir_absint',
                'ppg_red_ir_mean', 'ppg_red_ir_std'],
        'GSR': ['gsr_mean', 'gsr_std', 'gsr_var', 'gsr_median', 'gsr_min', 'gsr_max', 'gsr_range',
                'gsr_slope', 'gsr_tonic_mean', 'gsr_tonic_std', 'gsr_phasic_std'],
        'Temp': ['temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range',
                 'temp_var', 'temp_rms', 'temp_slope', 'temp_deriv_mean', 'temp_deriv_std']
    }

    slowFeatures = [
        'ecg_lf_power', 'ecg_hf_power', 'ecg_uhf_power', 'ecg_total_power',
        'ecg_lf_hf_ratio', 'ecg_rel_lf', 'ecg_rel_hf', 'ecg_rel_uhf',
        'ecg_rmssd', 'ecg_pnn50', 'ecg_nn50', 'ecg_rr_std',
        'ppg_rr_mean', 'ppg_rr_std', 'ppg_rr_amp_mean', 'ppg_rr_amp_std',
        'ppg_rmssd', 'ppg_pnn50', 'ppg_ibi_std',
        'gsr_tonic_mean', 'gsr_tonic_std', 'gsr_phasic_std', 'gsr_slope', 'gsr_mean',
        'temp_slope', 'temp_deriv_mean', 'temp_mean',
    ]

    contextSuffix = '__expmean'

    def __init__(self):
        self.accAnalyzer = AccAnalysis()
        self.ecgAnalyzer = ECGAnalysis()
        self.eegAnalyzer = EEGAnalysis()
        self.ppgAnalyzer = PPGAnalysis()
        self.gsrAnalyzer = GSRAnalysis()
        self.tempAnalyzer = TempAnalysis()

    def extractSessionFeatures(self, subjectData, sessionKey):
        featureDict = {}
        timestampDict = {}

        if 'eeg_ecg' in subjectData and sessionKey in subjectData['eeg_ecg']:
            data = subjectData['eeg_ecg'][sessionKey]
            fs = data['samplingRate']

            ecgPreprocessed = self.ecgAnalyzer.preprocessECG(np.array(data['ECG']), fs)
            featureDict['ECG'], timestampDict['ECG'] = self.ecgAnalyzer.extractFeatures(ecgPreprocessed, fs)

            eegPreprocessed = self.eegAnalyzer.preprocessEEG(np.array(data['EEG']), fs)
            featureDict['EEG'], timestampDict['EEG'] = self.eegAnalyzer.extractFeatures(eegPreprocessed, fs)

        if 'peripheral' in subjectData and sessionKey in subjectData['peripheral']:
            peripheralData = subjectData['peripheral'][sessionKey]
            if 'PPG' in peripheralData:
                fs = peripheralData['PPG']['samplingFreq']
                green = self.ppgAnalyzer.preprocessPPG(np.array(peripheralData['PPG']['dataCh1']), fs)
                red = self.ppgAnalyzer.preprocessPPG(np.array(peripheralData['PPG']['dataCh2']), fs)
                ir = self.ppgAnalyzer.preprocessPPG(np.array(peripheralData['PPG']['dataCh3']), fs)
                featureDict['PPG'], timestampDict['PPG'] = self.ppgAnalyzer.extractFeatures(green, red, ir, fs)

            if 'GSR' in peripheralData:
                fs = peripheralData['GSR']['samplingFreq']
                gsrPrep = self.gsrAnalyzer.preprocessGSR(np.array(peripheralData['GSR']['data']), fs)
                featureDict['GSR'], timestampDict['GSR'] = self.gsrAnalyzer.extractFeatures(gsrPrep, fs)

            if 'Temp' in peripheralData:
                fs = peripheralData['Temp']['samplingFreq']
                tempPrep = self.tempAnalyzer.preprocessTemp(np.array(peripheralData['Temp']['data']), fs)
                featureDict['Temp'], timestampDict['Temp'] = self.tempAnalyzer.extractFeatures(tempPrep, fs)

            if 'ACC' in peripheralData:
                fs = peripheralData['ACC']['samplingFreq']
                accData = np.column_stack([peripheralData['ACC']['x'], peripheralData['ACC']['y'], peripheralData['ACC']['z']])
                featureDict['ACC'], timestampDict['ACC'] = self.accAnalyzer.extractFeatures(accData, fs)

            df, timestamps = self.truncateAndCombine(featureDict, timestampDict)
            return self.addCausalContext(df), timestamps

    def truncateAndCombine(self, featureDict, timestampDict):
        if not featureDict:
            return pd.DataFrame(), np.array([])

        minWindows = min(len(t) for t in timestampDict.values())
        if minWindows == 0:
            return pd.DataFrame(), np.array([])

        refTimeStamps = list(timestampDict.values())[0][:minWindows]
        alignedData = {'timestamp': refTimeStamps}

        for modality, features in featureDict.items():
            truncated = features[:minWindows]
            names = self.featureNames.get(modality, [f'{modality}_{i}' for i in range(features.shape[1])])
            for j, name in enumerate(names):
                if j < truncated.shape[1]:
                    alignedData[name] = truncated[:, j]

        return pd.DataFrame(alignedData), np.array(refTimeStamps)

    def addCausalContext(self, df):
        if len(df) == 0:
            return df
        present = [name for name in self.slowFeatures if name in df.columns]
        context = {f'{name}{self.contextSuffix}': df[name].expanding().mean()
                   for name in present}
        return df.assign(**context) if context else df

    def extractSubjectFeatures(self, subjectData, subjectID):
        allDataFrames = []
        questionnaireData = subjectData.get('questionnaire', {})
        cumulativeTime = 0.0

        for sessionKey in self.sessionOrder:
            df, timestamps = self.extractSessionFeatures(subjectData, sessionKey)
            if len(df) == 0:
                print(f"Warning: no features for subject {subjectID} in session {sessionKey}")
                continue
            activity, condition = sessionKey.split('_')
            df['subject_id'] = subjectID
            df['activity'] = activity
            df['condition'] = condition
            df['session'] = sessionKey
            df['session_order'] = self.sessionOrder.index(sessionKey)
            df['condition_label'] = Config.condition_order.get(condition, -1)
            df['cumulative_time'] = df['timestamp'] + cumulativeTime
            cumulativeTime += df['timestamp'].max()
            if sessionKey in questionnaireData:
                df['PA'] = questionnaireData[sessionKey]['PA']
                df['NA'] = questionnaireData[sessionKey]['NA']
                df['SA'] = questionnaireData[sessionKey]['SA']
            allDataFrames.append(df)
        if not allDataFrames:
            return pd.DataFrame()

        result = pd.concat(allDataFrames, ignore_index=True)
        cols = result.columns.tolist()
        if 'cumulative_time' in cols:
            cols.remove('cumulative_time')
            cols.insert(1, 'cumulative_time')
            result = result[cols]
        return result

    def getAllFeatuerNames(self):
        names = []
        for modality in list(self.featureNames.keys()):
            names.extend(self.featureNames[modality])
        names.extend(f'{name}{self.contextSuffix}' for name in self.slowFeatures)
        return names

