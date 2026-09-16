import numpy as np
from scipy.signal import filtfilt
from Stress_study_ear_sum.BioSignalAnalysis.signalFilters import SignalFilters
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.BioSignalAnalysis.globalAnalysisHelper import GlobalHelper


class AccAnalysis(SignalFilters):
    def __init__(self):
        super().__init__()
        self.filterMethods = SignalFilters
        self.windowSec = Config.windowSize
        self.stepSec = Config.windowStep
        self.accLookBack = Config.AccLookBack
        self.notchFreq = Config.notchFreq
        self.helper = GlobalHelper()

    def preprocessAcc(self, data, fs, apply_notch=False):
        if apply_notch:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            processedData = filtfilt(b, a, data)
            return processedData
        return data

    def extractFeatures(self, data, fs):
        # data: (nSamples, 3) for x, y, z
        # feature matrix = (n_windows, n_features)
        windows, timestamps = self.helper.createSlidingWindows(data=data, sf=fs, lookbackSec=self.accLookBack, stepSec=self.stepSec, minStartSec=self.windowSec)
        featureMatrix = []
        for windowData in windows:
            features = self.computeWindowedFeatures(windowData, fs)
            featureMatrix.append(features)

        return np.array(featureMatrix), timestamps

    def computeWindowedFeatures(self, windowData, fs):
        # windowed data: (nSamples, 3) for x, y, z
        if windowData.ndim == 1:
            x = windowData
            y = np.zeros_like(x)
            z = np.zeros_like(x)
        else:
            x = windowData[:, 0]
            y = windowData[:, 1]
            z = windowData[:, 2]

        features = []
        # statistical features
        magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        features.extend([np.mean(x), np.mean(y), np.mean(z), np.mean(magnitude)])
        features.extend([np.std(x), np.std(y), np.std(z), np.std(magnitude)])

        # absolute integral
        features.extend([np.sum(np.abs(x)), np.sum(np.abs(y)), np.sum(np.abs(z)), np.sum(np.abs(magnitude))])

        # peak frequency
        for axisData in [x, y, z]:
            n = len(axisData)
            if n > 1:
                fftVals = np.abs(np.fft.rfft(axisData))
                freqs = np.fft.rfftfreq(n, 1/fs)
                peakIdx = np.argmax(fftVals[1:]) + 1 if len(fftVals) > 1 else 0
                features.append(freqs[peakIdx] if peakIdx < len(freqs) else 0)
            else:
                features.append(0.0)

        return features







