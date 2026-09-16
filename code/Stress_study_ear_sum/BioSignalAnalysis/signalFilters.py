import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


class SignalFilters:
    def __init__(self):
        pass

    @staticmethod
    def butter_bandpass(cutoffFreqs, fs, order=4):
        nyquist = 0.5 * fs
        low = cutoffFreqs[0] / nyquist
        high = cutoffFreqs[1] / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos

    @staticmethod
    def butter_lowpass(cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='low', output='sos')
        return sos

    @staticmethod
    def butter_highpass(cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='high', output='sos')
        return sos

    @staticmethod
    def notch_filter(freq, fs, Q=30):
        nyquist = 0.5 * fs
        if freq >= nyquist:
            return np.array([1.0]), np.array([1.0])
        w0 = freq / nyquist
        b, a = iirnotch(w0, Q)
        return b, a

    @staticmethod
    def remove_outliers_iqr(features, k=1.5):
        if features.size == 0:
            return features
        cleaned_features = features.copy()
        for col_idx in range(features.shape[1]):
            column = features[:, col_idx]
            Q1 = np.percentile(column, 25)
            Q3 = np.percentile(column, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            outlier_mask = (column < lower_bound) | (column > upper_bound)
            if np.any(outlier_mask):
                median_val = np.median(column[~outlier_mask])
                cleaned_features[outlier_mask, col_idx] = median_val
        return cleaned_features
