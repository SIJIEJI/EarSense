import numpy as np
import neurokit2 as nk
from scipy.signal import filtfilt, sosfiltfilt, welch

from Stress_study_ear_sum.BioSignalAnalysis.signalFilters import SignalFilters
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.BioSignalAnalysis.globalAnalysisHelper import GlobalHelper


class ECGAnalysis(SignalFilters):
    def __init__(self):
        super().__init__()
        self.filterMethods = SignalFilters
        self.cutOffFreqs = Config.ECGBandPass
        self.notchFreq = Config.notchFreq
        self.windowSec = Config.windowSize
        self.stepSec = Config.windowStep
        self.ecgLookBack = Config.ECGLookBack

        self.helper = GlobalHelper()

    def preprocessECG(self, data, fs, apply_notch=True):
        if apply_notch:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            filteredData = filtfilt(b, a, data)
            sos = self.filterMethods.butter_bandpass(self.cutOffFreqs, fs, order=4)
            filteredData = sosfiltfilt(sos, filteredData)
        else:
            sos = self.filterMethods.butter_bandpass(self.cutOffFreqs, fs, order=4)
            filteredData = sosfiltfilt(sos, data)
        return filteredData

    def extractFeatures(self, data, fs):
        windows, timestamps = self.helper.createSlidingWindows(data=data, sf=fs, lookbackSec=self.ecgLookBack, stepSec=self.stepSec, minStartSec=self.windowSec)
        featureMatrix = []
        for windowData in windows:
            features = self.computeWindowedFeatures(windowData, fs)
            featureMatrix.append(features)
        return np.array(featureMatrix), timestamps

    def detectRPeak(self, ecgData, fs):
        # additional butter filter
        ecgCleaned = nk.ecg_clean(ecgData, fs)
        signals, info = nk.ecg_peaks(ecgCleaned, fs, method='neurokit')
        RPeaks = info['ECG_R_Peaks']
        return RPeaks

    def computeWindowedFeatures(self, windowData, fs):
        features = []
        rPeaks = self.detectRPeak(windowData, fs)
        if len(rPeaks) < 2:
            return [0.0] * 32
        rrIntervals = np.diff(rPeaks) / fs * 1000
        hr = 60000 / rrIntervals
        features.extend([np.mean(hr), np.std(hr), np.mean(rrIntervals), np.std(rrIntervals)])
        rrDiff = np.diff(rrIntervals)
        rmssd = np.sqrt(np.mean(rrDiff**2)) if len(rrDiff) > 0 else 0.0
        features.append(rmssd)
        nn50 = np.sum(np.abs(rrDiff) > 50) if len(rrDiff) > 0 else 0
        pnn50 = (nn50 / len(rrDiff)) * 100 if len(rrDiff) > 0 else 0
        features.extend([nn50, pnn50])
        freqFeatures = self.computeFrequencyFeatures(rrIntervals)
        features.extend(freqFeatures)
        features = [0.0 if np.isnan(f) else f for f in features]
        return features

    def computeFrequencyFeatures(self, rrIntervals):
        features = []
        if len(rrIntervals) < 4:
            return [0.0] * 12
        rrTimes = np.cumsum(rrIntervals) / 1000
        # resample to 4hz
        fsResample = 4.0
        tInterp = np.arange(0, rrTimes[-1], 1/fsResample)
        if len(tInterp) < 4:
            return [0.0] * 10
        rrInterp = np.interp(tInterp, rrTimes, rrIntervals)
        rrInterp = rrInterp - np.mean(rrInterp)
        nperseg = min(256, len(rrInterp))
        freqs, psd = welch(rrInterp, fs=fsResample, nperseg=nperseg)
        # frequency bands
        bands = {
            'LF': (0.04, 0.15),
            'HF': (0.15, 0.4),
            'UHF': (0.4, 1.0)
        }
        bandPowers = {}
        for bandName, (lowF, highF) in bands.items():
            idx = np.where((freqs >= lowF) & (freqs <= highF))[0]
            if len(idx) > 1:
                bandPowers[bandName] = np.trapz(psd[idx], freqs[idx])
            else:
                bandPowers[bandName] = 0.0

        lfPower = bandPowers['LF']
        hfPower = bandPowers['HF']
        uhfPower = bandPowers['UHF']
        totalPower = lfPower + hfPower + uhfPower
        features.extend([ lfPower, hfPower, uhfPower, totalPower])
        lfhfRatio = lfPower / hfPower if hfPower > 0 else 0.0
        relLFPower = lfPower / totalPower if totalPower > 0 else 0.0
        relHFPower = hfPower / totalPower if totalPower > 0 else 0.0
        relUHFPower = uhfPower / totalPower if totalPower > 0 else 0.0
        features.extend([lfhfRatio, relLFPower, relHFPower, relUHFPower])
        return features


