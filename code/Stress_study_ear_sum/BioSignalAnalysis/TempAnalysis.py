import numpy as np
from scipy.signal import filtfilt
from Stress_study_ear_sum.BioSignalAnalysis.signalFilters import SignalFilters
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.BioSignalAnalysis.globalAnalysisHelper import GlobalHelper


class TempAnalysis(SignalFilters):
    def __init__(self):
        super().__init__()
        self.filterMethods = SignalFilters
        self.notchFreq = Config.notchFreq
        self.filterMethods = SignalFilters
        self.windowSec = Config.windowSize
        self.stepSec = Config.windowStep
        self.tempLookBack = Config.TempLookBack
        self.helper = GlobalHelper()

    def preprocessTemp(self, data, fs, apply_notch=False):
        if apply_notch:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            processedData = filtfilt(b, a, data)
            return processedData
        return data

    def extractFeatures(self, data, fs):
        windows, timestamps = self.helper.createSlidingWindows(data=data, sf=fs, lookbackSec=self.tempLookBack, stepSec=self.stepSec, minStartSec=self.windowSec)
        featureMatrix = []
        for windowData in windows:
            features = self.computeWindowedFeatures(windowData, fs)
            featureMatrix.append(features)
        return np.array(featureMatrix), timestamps

    def computeWindowedFeatures(self, windowData, fs):
        features = []
        mean = np.mean(windowData)
        std = np.std(windowData)
        min = np.min(windowData)
        max = np.max(windowData)
        range = np.ptp(windowData)
        var = np.var(windowData)
        rms = np.sqrt(np.mean(windowData ** 2))
        x = np.arange(len(windowData))
        slope, _ = np.polyfit(x, windowData, 1)
        deriv = np.diff(windowData) * fs
        firstDerivMean = np.mean(deriv)
        firstDerivStd = np.std(deriv)
        features.extend([mean, std, min, max, range, var, rms, slope, firstDerivMean, firstDerivStd])
        features = [0.0 if np.isnan(x) else x for x in features]
        return features
