import numpy as np
from scipy.signal import filtfilt, sosfiltfilt

from Stress_study_ear_sum.BioSignalAnalysis.signalFilters import SignalFilters
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.BioSignalAnalysis.globalAnalysisHelper import GlobalHelper


class GSRAnalysis(SignalFilters):
    def __init__(self):
        super().__init__()
        self.filterMethods = SignalFilters
        self.notchFreq = Config.notchFreq
        self.windowSec = Config.windowSize
        self.stepSec = Config.windowStep
        self.gsrLookBack = Config.GSRLookBack
        self.helper = GlobalHelper()

    def preprocessGSR(self, data, fs, apply_notch=False):
        if apply_notch:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            processedData = filtfilt(b, a, data)
            return processedData
        return data

    def extractFeatures(self, data, fs):
        windows, timestamps = self.helper.createSlidingWindows(data=data, sf=fs, lookbackSec=self.gsrLookBack, stepSec=self.stepSec, minStartSec=self.windowSec)
        featureMatrix = []
        for windowData in windows:
            features = self.computeWindowedFeatures(windowData, fs)
            featureMatrix.append(features)
        return np.array(featureMatrix), timestamps

    def decomposeSCLandSCR(self, data, fs, movingAvgWindowSize=4):
        windowSize = int(movingAvgWindowSize*fs)
        if windowSize > len(data):
            windowSize = len(data)
        if windowSize < 1:
            windowSize = 1

        scl = np.convolve(data, np.ones(windowSize)/windowSize, mode='same')
        scr = data - scl
        return scl, scr

    def computeWindowedFeatures(self, windowData, fs):
        features = []
        features.extend([np.mean(windowData), np.std(windowData), np.var(windowData), np.median(windowData), np.min(windowData), np.max(windowData), np.ptp(windowData)])
        x = np.arange(len(windowData))
        slope, _ = np.polyfit(x, windowData, 1)
        features.append(slope)
        scl, scr = self.decomposeSCLandSCR(windowData, fs)
        tonicMean = np.mean(scl)
        tonicStd = np.std(scl)
        phasicStd = np.std(scr)
        features.extend([tonicMean, tonicStd, phasicStd])

        features = [0.0 if np.isnan(f) else f for f in features]
        return features


