import numpy as np
from scipy import signal, stats
from scipy.signal import filtfilt, sosfiltfilt, welch

from Stress_study_ear_sum.BioSignalAnalysis.signalFilters import SignalFilters
from Stress_study_ear_sum.config import Config
from Stress_study_ear_sum.BioSignalAnalysis.globalAnalysisHelper import GlobalHelper


class EEGAnalysis(SignalFilters):
    def __init__(self):
        super().__init__()
        self.filterMethods = SignalFilters
        self.notchFreq = Config.notchFreq
        self.cutOffFreqs = Config.EEGBandPass
        self.windowSec = Config.windowSize
        self.stepSec = Config.windowStep
        self.eegLookBack = Config.EEGLookBack
        self.helper = GlobalHelper()
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    def preprocessEEG(self, data, fs, apply_notch=True):
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
        windows, timestamps = self.helper.createSlidingWindows(data=data, sf=fs, lookbackSec=self.eegLookBack, stepSec=self.stepSec, minStartSec=self.windowSec)
        featureMatrix = []
        for windowData in windows:
            features = self.computeWindowedFeatures(windowData, fs)
            featureMatrix.append(features)
        return np.array(featureMatrix), timestamps

    def computeHjorthParameters(self, data):
        activity = np.var(data)
        firstDerivative = np.diff(data)
        secondDerivative = np.diff(firstDerivative)
        varianceFirstDeriv = np.var(firstDerivative)
        varianceSecondDeriv = np.var(secondDerivative)
        if activity > 0:
            mobility = np.sqrt(varianceFirstDeriv / activity)
        else:
            mobility = 0.0
        if varianceFirstDeriv > 0:
            mobilityFirstDeriv = np.sqrt(varianceSecondDeriv / varianceFirstDeriv)
        else:
            mobilityFirstDeriv = 0.0
        if mobility > 0:
            complexity = mobilityFirstDeriv / mobility
        else:
            complexity = 0.0
        return activity, mobility, complexity

    def computeBandPowers(self, data, fs):
        nperseg = min(len(data), int(4 * fs))
        freqs, psd = welch(data, fs=fs, nperseg=nperseg)
        bandPowers = {}
        for bandName, (lowFreq, highFreq) in self.bands.items():
            idx = np.where((freqs >= lowFreq) & (freqs <= highFreq))[0]
            if len(idx) > 1:
                bandPowers[bandName] = np.trapz(psd[idx], freqs[idx])
            else:
                bandPowers[bandName] = 0.0
        totalPower = np.sum(list(bandPowers.values()))
        relPowers = {}
        for bandName, power in bandPowers.items():
            relPowers[bandName] = power / totalPower if totalPower > 0 else 0.0

        return bandPowers, relPowers, totalPower, freqs, psd

    def computeSpectralFeatures(self, freqs, psd):
        psdNorm = psd / np.sum(psd) if np.sum(psd) > 0 else 0.0
        psdNorm = psdNorm[psdNorm > 0] # remove 0 for log
        spectralEntropy = -np.sum(psdNorm * np.log2(psdNorm)) if len(psdNorm) > 0 else 0.0
        meanFreq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0.0
        peakFreq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0

        return spectralEntropy, meanFreq, peakFreq

    def computeWindowedFeatures(self, windowData, fs):
        features = []
        activity, mobility, complexity = self.computeHjorthParameters(windowData)
        features.extend([activity, mobility, complexity])
        bandPowers, relPowers, totalPower, freqs, psd = self.computeBandPowers(windowData, fs)
        alpha = bandPowers['alpha']
        beta = bandPowers['beta']
        theta = bandPowers['theta']
        delta = bandPowers['delta']
        gamma = bandPowers['gamma']

        relDelta = relPowers['delta']
        relTheta = relPowers['theta']
        relAlpha = relPowers['alpha']
        relBeta = relPowers['beta']
        relGamma = relPowers['gamma']
        features.extend([delta, theta, alpha, beta, gamma, totalPower])
        features.extend([relDelta, relTheta, relAlpha, relBeta, relGamma])

        alphaBetaRatio = alpha / beta if beta > 0 else 0.0
        thetaAlphaRatio = theta / alpha if alpha > 0 else 0.0
        thetaBetaRatio = theta / beta if beta > 0 else 0.0
        alphaThetaRatio = alpha / theta if theta > 0 else 0.0
        betaAlphaRatio = beta / alpha if alpha > 0 else 0.0  # stress increase this
        deltaBetaRatio = delta / beta if beta > 0 else 0.0
        deltaAlphaRatio = delta / alpha if alpha > 0 else 0.0
        alphaThetaSum = alpha + theta
        engagementIndex = beta / alphaThetaSum if alphaThetaSum > 0 else 0.0
        features.extend([alphaBetaRatio, thetaAlphaRatio, thetaBetaRatio, alphaThetaRatio, betaAlphaRatio, deltaBetaRatio, deltaAlphaRatio, engagementIndex])

        spectralEntropy, meanFreq, peakFreq = self.computeSpectralFeatures(freqs, psd)
        features.extend([spectralEntropy, meanFreq, peakFreq])
        features.extend([np.mean(windowData), np.std(windowData), np.var(windowData), np.min(windowData), np.max(windowData), np.ptp(windowData)])

        rms = np.sqrt(np.mean(windowData ** 2))
        skewness = stats.skew(windowData)
        kurtosis = stats.kurtosis(windowData)
        features.extend([rms, skewness, kurtosis])
        zeroCrossings = np.sum(np.diff(np.sign(windowData)) != 0)
        features.append(zeroCrossings / len(windowData))

        features = [0.0 if np.isnan(f) else f for f in features]
        return features

