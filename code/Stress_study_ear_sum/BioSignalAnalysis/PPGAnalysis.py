import numpy as np
import neurokit2 as nk
from scipy.signal import filtfilt, sosfiltfilt, welch, find_peaks

from Stress_study_ear_sum.BioSignalAnalysis.signalFilters import SignalFilters
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.BioSignalAnalysis.globalAnalysisHelper import GlobalHelper


class PPGAnalysis(SignalFilters):
    def __init__(self):
        super().__init__()
        self.filterMethods = SignalFilters
        self.notchFreq = Config.notchFreq
        self.cutOffFreqsRedIR = Config.PPGIRBandPass if Config.PPGRedBandPass is None else Config.PPGRedBandPass
        self.breathCutoff = Config.PPGGreenRespBandPass
        self.pulseCutoff = Config.PPGGreenPulseBandPass
        self.windowSec = Config.windowSize
        self.stepSec = Config.windowStep
        self.ppgLookBack = Config.PPGLookBack
        self.helper = GlobalHelper()

    def preprocessPPG(self, data, fs, apply_notch=True):
        if apply_notch and fs > 2 * self.notchFreq:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            filteredData = filtfilt(b, a, data)
        else:
            filteredData = data.copy()
        sos = self.filterMethods.butter_bandpass(self.cutOffFreqsRedIR, fs, order=4)
        filteredData = sosfiltfilt(sos, filteredData)
        return filteredData

    def preprocessPPGPulse(self, data, fs, apply_notch=True):
        if apply_notch:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            filteredData = filtfilt(b, a, data)
        else:
            filteredData = data.copy()
        sos = self.filterMethods.butter_bandpass(self.pulseCutoff, fs, order=4)
        filteredData = sosfiltfilt(sos, filteredData)
        return filteredData

    def preprocessPPGResp(self, data, fs, apply_notch=True):
        if apply_notch:
            b, a = self.filterMethods.notch_filter(self.notchFreq, fs, Q=30)
            filteredData = filtfilt(b, a, data)
        else:
            filteredData = data.copy()
        sos = self.filterMethods.butter_bandpass(self.breathCutoff, fs, order=4)
        filteredData = sosfiltfilt(sos, filteredData)
        return filteredData

    def extractFeatures(self, greenData, redData, irData, fs):
        self.helper = GlobalHelper()
        allData = np.column_stack([greenData, redData, irData])

        windows, timestamps = self.helper.createSlidingWindows(data=allData, sf=fs, lookbackSec=self.ppgLookBack, stepSec=self.stepSec, minStartSec=self.windowSec)
        featureMatrix = []
        for windowData in windows:
            features = self.computeWindowedFeatures(windowData, fs)
            featureMatrix.append(features)
        return np.array(featureMatrix), timestamps

    def detectPulse(self, ppgData, fs):
        ppgCleaned = nk.ppg_clean(ppgData, sampling_rate=fs)
        info = nk.ppg_findpeaks(ppgCleaned, sampling_rate=fs, method='elgendi')
        peaks = info['PPG_Peaks']
        return np.array(peaks)

    def detectRespiration(self, respData, fs):
        minDistance = int(1.5 * fs)
        peaks, _ = find_peaks(respData, distance=minDistance)
        return peaks

    def computeWindowedFeatures(self, windowData, fs):
        features = []
        green = windowData[:, 0]
        red = windowData[:, 1]
        ir = windowData[:, 2]

        greenPulse = self.preprocessPPG(green, fs, apply_notch=True)
        pulsePeaks = self.detectPulse(greenPulse, fs)
        if len(pulsePeaks) >= 2:
            ibi = np.diff(pulsePeaks) / fs * 1000
            pulseRate = 60000 / ibi
            features.extend([np.mean(pulseRate), np.std(pulseRate), np.min(pulseRate), np.max(pulseRate), np.ptp(pulseRate)])
            features.extend([np.mean(ibi), np.std(ibi)])

            ibiDiff = np.diff(ibi)
            rmssd = np.sqrt(np.mean(ibiDiff**2)) if len(ibiDiff) > 0 else 0.0
            features.append(rmssd)

            nn50 = np.sum(np.abs(ibiDiff) > 50) if len(ibiDiff) > 0 else 0
            pnn50 = (nn50 / len(ibiDiff) * 100) if len(ibiDiff) > 0 else 0.0
            features.append(pnn50)
        else:
            features.extend([0.0] * 9)

        greenResp = self.preprocessPPGResp(green, fs, apply_notch=False)
        respPeaks = self.detectRespiration(greenResp, fs)

        if len(respPeaks) >= 2:
            breathIntervals = np.diff(respPeaks) / fs
            respRate = 60 / breathIntervals
            respAmps = greenResp[respPeaks]
            features.extend([np.mean(respRate), np.std(respRate), np.mean(respAmps), np.std(respAmps)])
        else:
            features.extend([0.0] * 4)

        features.extend([np.mean(green), np.std(green), np.min(green), np.max(green), np.ptp(green)])
        features.extend([np.mean(red), np.std(red), np.min(red), np.max(red), np.ptp(red), np.sum(np.abs(red))])
        features.extend([np.mean(ir), np.std(ir), np.min(ir), np.max(ir), np.ptp(ir), np.sum(np.abs(ir))])

        irSafe = np.where(ir !=0, ir, 1e-10)
        redIrRatio = red / irSafe
        features.extend([np.mean(redIrRatio), np.std(redIrRatio)])
        features = [0.0 if np.isnan(f) else f for f in features]

        return features




