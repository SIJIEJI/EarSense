import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, peak_prominences
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import savgol_filter



class ppgAnalysisProtocol:
    def __init__(self, ratioBounds, greenCutOff, irCutOff, redCutOff, breathCutOff, lowPassFreqs, sf=100):
        self.sf = sf

        self.greenCutOff = greenCutOff # general band pass for heart rate green signal; and IR and red spo2
        self.irCutOff = irCutOff
        self.redCutOff = redCutOff
        self.breathCutOff = breathCutOff # band pass for green to calculate breath rete
        self.lowPassFreq = lowPassFreqs # low pass for dc computation for IR and red

        # Artifact rejection parameters
        self.max_zscore_peak_heartbeat = 2.0  # threshold for heartbeat
        self.max_zscore_peak_breathing = 2.0  # threshold for breathing
        self.max_zscore_peak_spo2 = 2.0
        self.ratioBounds = ratioBounds


    def bandpass_filter(self, signal, cutoffFreqs, order=3):
        nyq = 0.5 * self.sf
        normalCutoff = np.asarray(cutoffFreqs)/nyq
        sos = butter(order, normalCutoff, btype='band', output='sos')
        filtered_signal = sosfiltfilt(sos, signal)
        return filtered_signal

    def fft_bandpass_filter(self, signal, sampling_rate, low_cut, high_cut):
        """
        Apply an FFT-based bandpass filter to a signal.

        This function computes the FFT of the input signal, retains only those
        frequency components between low_cut and high_cut, and then reconstructs the
        time-domain signal using the inverse FFT.

        Parameters:
            signal (array-like): Input time-domain signal.
            sampling_rate (float): Sampling rate of the signal in Hz.
            low_cut (float): Low cutoff frequency (Hz). Frequencies below this value will be suppressed.
            high_cut (float): High cutoff frequency (Hz). Frequencies above this value will be suppressed.

        Returns:
            np.ndarray: The filtered signal (real part).
        """
        # Compute the FFT of the signal
        fft_signal = np.fft.fft(signal)
        N = len(signal)

        # Create a frequency axis corresponding to FFT bins
        freqs = np.fft.fftfreq(N, d=1 / sampling_rate)

        # Create a mask to keep only the frequencies in the desired band
        mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)

        # Zero out FFT components outside the band
        fft_signal_filtered = fft_signal * mask

        # Inverse FFT to convert back to time domain
        filtered_signal = np.fft.ifft(fft_signal_filtered)

        # Due to numerical errors, the imaginary part should be negligible,
        # so we only return the real component.
        return np.real(filtered_signal)


    def lowpass_filter(self, signal, cutoffFreq, order=3):
        nyq = 0.5 * self.sf
        normalCutoff = np.asarray(cutoffFreq)/nyq
        sos = butter(order, normalCutoff, btype='low', output='sos')
        filtered_signal = sosfiltfilt(sos, signal)
        return filtered_signal

    def standardize_signal(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal  # prevent divide-by-zero
        standardized = (signal - mean) / std
        return standardized

    def is_good_signal(self, signal, max_allowed_peak=3.0):
        standardized_signal = self.standardize_signal(signal)
        max_peak = np.max(np.abs(standardized_signal))
        return max_peak <= max_allowed_peak

    def heartRateCalculation(self, heartBeatSignal, plot=False, removeWindow=False):
        min_distance = int(0.4 * self.sf)
        peak_prominence = np.std(heartBeatSignal)/4 # (np.max(heartBeatSignal) - np.min(heartBeatSignal)) * 0.01

        peaks, _ = find_peaks(
            heartBeatSignal,
            distance=min_distance,
            prominence=peak_prominence
        )

        if len(peaks) < 2:
            return np.nan

        # Standardize the signal
        standardized_signal = self.standardize_signal(heartBeatSignal)
        if removeWindow:
            if np.std(standardized_signal) > self.max_zscore_peak_heartbeat:
                print("Signal too noisy, skipping window.")
                return np.nan


        if plot:
            t = np.arange(len(heartBeatSignal)) / self.sf
            plt.figure(figsize=(10, 4))
            plt.plot(t, heartBeatSignal, label="HeartBeat Signal", color="#1f78b4")
            plt.plot(t[peaks], heartBeatSignal[peaks], 'rx', label="Good Peaks", markersize=8)
            plt.title("Detected Heartbeat Peaks (Hard + Zscore Filtered)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()

        if len(peaks) < 2:
            return np.nan  # not enough good peaks

        rwaveIntervals = np.diff(peaks) / self.sf
        avgRwave = np.mean(rwaveIntervals)
        heartRate = 60 / avgRwave

        return heartRate

    def breathingRateCalculation(self, breathingSignal, plot=False, removeWindow=False):
        min_distance = int(2 * self.sf)
        peak_prominence = np.std(breathingSignal)/4 #(np.max(breathingSignal) - np.min(breathingSignal)) * 0.05

        peaks, _ = find_peaks(
            breathingSignal,
            distance=min_distance,
            prominence=peak_prominence
        )

        if len(peaks) < 2:
            return np.nan

        # Standardize the breathing signal
        standardized_signal = self.standardize_signal(breathingSignal)
        if removeWindow:
            if np.std(standardized_signal) > self.max_zscore_peak_breathing:
                print("Signal too noisy, skipping window.")
                return np.nan

        if plot:
            t = np.arange(len(breathingSignal)) / self.sf
            plt.figure(figsize=(10, 4))
            plt.plot(t, breathingSignal, label="Breathing Signal", color="#33a02c")
            plt.plot(t[peaks], breathingSignal[peaks], 'rx', label="Good Peaks", markersize=8)
            plt.title("Detected Breathing Peaks (Hard + Zscore Filtered)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()

        if len(peaks) < 2:
            return np.nan

        intervals = np.diff(peaks) / self.sf
        avgInterval = np.mean(intervals)
        breathingRate = 60 / avgInterval

        return breathingRate

    def validRatioCheck(self, ratio):
        if ratio < self.ratioBounds[0]:
            return self.ratioBounds[0]
        if ratio > self.ratioBounds[1]:
            return self.ratioBounds[1]
        else:
            return ratio


    # def calculateSP02_Separate(self, irLow, redLow, irFilteredSignal, redFilteredSignal, heart_rate=None, plot=False):

    #     if heart_rate is not None and not np.isnan(heart_rate) and heart_rate > 0: # just in case
    #         rr_interval = 60.0 / heart_rate
    #         dynamic_window = int(rr_interval * self.sf * 0.8)
    #     else:
    #         dynamic_window = int(0.8 * self.sf)

    #     # Take mean of smoothed DC components
    #     irDC = np.mean(irLow)
    #     redDC = np.mean(redLow)
    #     window_size = int(0.4 * self.sf)

    #     irPeaks, _ = find_peaks(irFilteredSignal, distance=window_size, prominence=np.std(irFilteredSignal)/4)
    #     redPeaks, _ = find_peaks(redFilteredSignal, distance=window_size, prominence=np.std(redFilteredSignal)/4)

    #     # Initialize empty lists for good peaks.
    #     ir_good_peaks = []
    #     red_good_peaks = []
    #     # Standardize the signals.
    #     standardized_irsignal = self.standardize_signal(irFilteredSignal)
    #     standardized_redsignal = self.standardize_signal(redFilteredSignal)
    #     # Iterate over each IR peak individually.
    #     for irpeak in irPeaks:
    #         if np.abs(standardized_irsignal[irpeak]) <= self.max_zscore_peak_spo2:
    #             ir_good_peaks.append(irpeak)
    #     # Iterate over each red peak individually.
    #     for redpeak in redPeaks:
    #         if np.abs(standardized_redsignal[redpeak]) <= self.max_zscore_peak_spo2:
    #             red_good_peaks.append(redpeak)
    #     # Convert the lists to numpy arrays.
    #     ir_good_peaks = np.array(ir_good_peaks)
    #     red_good_peaks = np.array(red_good_peaks)

    def calculateSP02_Separate(self, irLow, redLow, irFilteredSignal, redFilteredSignal, heart_rate=None, plot=False):
        irLow   = np.asarray(irLow, dtype=float)
        redLow  = np.asarray(redLow, dtype=float)
        irACsig = np.asarray(irFilteredSignal, dtype=float)
        rdACsig = np.asarray(redFilteredSignal, dtype=float)

        n = min(irLow.size, redLow.size, irACsig.size, rdACsig.size)
        if n < max(int(2 * self.sf), 50):
            return np.nan, np.nan
        irLow, redLow, irACsig, rdACsig = irLow[:n], redLow[:n], irACsig[:n], rdACsig[:n]

        mask = np.isfinite(irLow) & np.isfinite(redLow) & np.isfinite(irACsig) & np.isfinite(rdACsig)
        if mask.sum() < 10:
            return np.nan, np.nan
        irLow, redLow, irACsig, rdACsig = irLow[mask], redLow[mask], irACsig[mask], rdACsig[mask]

        # Dynamic window from HR (fallback 0.8 s)
        if heart_rate is not None and np.isfinite(heart_rate) and heart_rate > 0:
            rr = 60.0 / heart_rate
            dyn_w = int(max(0.5, 0.8 * rr) * self.sf)
        else:
            dyn_w = int(0.8 * self.sf)

        # Peak picking (heartbeat band already)
        min_dist = max(1, int(0.3 * self.sf))
        prom_ir  = max(np.std(irACsig) / 4.0, 1e-12)
        prom_rd  = max(np.std(rdACsig) / 4.0, 1e-12)
        ir_peaks, _ = find_peaks(irACsig, distance=min_dist, prominence=prom_ir)
        rd_peaks, _ = find_peaks(rdACsig, distance=min_dist, prominence=prom_rd)

        # Z-score culling
        std_ir = np.std(irACsig) or 1.0
        std_rd = np.std(rdACsig) or 1.0
        ir_good = ir_peaks[np.abs(irACsig[ir_peaks] / std_ir) <= self.max_zscore_peak_spo2]
        rd_good = rd_peaks[np.abs(rdACsig[rd_peaks] / std_rd) <= self.max_zscore_peak_spo2]

        # Per-beat AC amplitudes
        irAC = self._extract_ac_components(irACsig, ir_good, dyn_w, plot=False)
        rdAC = self._extract_ac_components(rdACsig, rd_good, dyn_w, plot=False)

        irAC = self._reject_outliers(irAC, z_thresh=3.0)
        rdAC = self._reject_outliers(rdAC, z_thresh=3.0)

        if irAC.size == 0 or rdAC.size == 0:
            return np.nan, np.nan

        # DC via median (robust)
        dc_ir  = float(np.nanmedian(irLow))
        dc_rd  = float(np.nanmedian(redLow))
        if not np.isfinite(dc_ir) or not np.isfinite(dc_rd) or abs(dc_ir) < 1e-9 or abs(dc_rd) < 1e-9:
            return np.nan, np.nan

        # Mean AC per window
        ac_ir = float(np.nanmean(irAC))
        ac_rd = float(np.nanmean(rdAC))
        if ac_ir <= 0 or ac_rd <= 0 or not np.isfinite(ac_ir) or not np.isfinite(ac_rd):
            return np.nan, np.nan

        ratio = (ac_rd / dc_rd) / (ac_ir / dc_ir)
        if not np.isfinite(ratio) or ratio <= 0:
            return np.nan, np.nan
        ratio = self.validRatioCheck(ratio)

        # Ear-PPG default; adjust as needed
        spo2 = 109.0 - 22.0 * ratio
        spo2 = float(np.clip(spo2, 70.0, 100.0))

        if plot:
            t = np.arange(n) / float(self.sf)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 3))
            plt.plot(t, irACsig, label="IR filtered")
            plt.plot(t[ir_good], irACsig[ir_good], 'rx', label="IR peaks")
            plt.grid(True, ls="--", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

            plt.figure(figsize=(10, 3))
            plt.plot(t, rdACsig, label="Red filtered", color="#ff7f0e")
            plt.plot(t[rd_good], rdACsig[rd_good], 'rx', label="Red peaks")
            plt.grid(True, ls="--", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

        return spo2, float(ratio)

    def _extract_ac_components(self, signal, peaks, dynamic_window, plot=False):
        ac = []
        w = int(dynamic_window)  # samples

        for i, peak in enumerate(peaks):
            start = max(0, peak - w // 2)
            end   = min(len(signal), peak + w // 2)
            segment = signal[start:end]
            if len(segment) < 3:
                continue

            # troughs before/after the peak (split segment in half)
            half = max(1, len(segment) // 2)
            trough_before = np.argmin(segment[:half])
            trough_after  = np.argmin(segment[half:]) + half
            baseline = min(segment[trough_before], segment[trough_after])
            amp = signal[peak] - baseline
            if amp > 0:
                ac.append(amp)

            if plot:
                t_seg = np.arange(start, end) / float(self.sf)
                plt.figure(figsize=(8, 3))
                plt.plot(t_seg, segment, label="Segment")
                plt.axvline(x=peak / self.sf, color='r', ls='--', label='Peak')
                plt.axvline(x=(start + trough_before) / self.sf, color='g', ls='--', label='Trough before')
                plt.axvline(x=(start + trough_after)  / self.sf, color='m', ls='--', label='Trough after')
                plt.title(f"Pulse {i+1}")
                plt.xlabel("Time (s)"); plt.grid(True, ls="--", alpha=0.5); plt.legend(); plt.tight_layout()
                plt.show()

        return np.asarray(ac, dtype=float)

    # def extract_AC_components(signal, peaks, dynamicWindow, plot=False):
    #     ac_amplitudes = []
    #     window = dynamicWindow  # I found 800ms covers the full range

    #     for i, peak in enumerate(peaks):
    #         start = max(0, peak - window // 2)
    #         end = min(len(signal), peak + window // 2)
    #         segment = signal[start:end]

    #         # If the segment is too short, skip it.
    #         if len(segment) < 3:
    #             continue

    #         # Identify the trough before the peak (using the first half of the segment)
    #         trough_before = np.argmin(segment[:window // 2])
    #         # Identify the trough after the peak (using the second half of the segment, offset by window//2)
    #         trough_after = np.argmin(segment[window // 2:]) + window // 2
    #         # Choose the lower of the two troughs as the baseline
    #         min_val = min(segment[trough_before], segment[trough_after])
    #         # Compute the amplitude as the difference between the peak and the effective trough.
    #         amplitude = signal[peak] - min_val

    #         if amplitude > 0:
    #             ac_amplitudes.append(amplitude)

    #         # Plot the segment and mark the peak and troughs if plot is True.
    #         if plot:
    #             t_segment = np.arange(start, end) / self.sf  # time axis in seconds
    #             plt.figure(figsize=(8, 4))
    #             plt.plot(t_segment, segment, label="Segment")
    #             # Mark the detected peak
    #             plt.axvline(x=peak / self.sf, color='red', linestyle='--', label='Peak')
    #             # Mark the trough before the peak
    #             plt.axvline(x=(start + trough_before) / self.sf, color='green', linestyle='--', label='Trough before')
    #             # Mark the trough after the peak
    #             plt.axvline(x=(start + trough_after) / self.sf, color='purple', linestyle='--', label='Trough after')
    #             plt.title(f"Pulse Segment {i + 1}")
    #             plt.xlabel("Time (s)")
    #             plt.legend()
    #             plt.grid(True, linestyle="--", alpha=0.5)
    #             plt.tight_layout()
    #             plt.show()

    #     return np.array(ac_amplitudes)

    #     irAC = extract_AC_components(irFilteredSignal, ir_good_peaks, dynamic_window, plot=False)
    #     redAC = extract_AC_components(redFilteredSignal, red_good_peaks, dynamic_window, plot=False)

        # Outlier rejection
    # def reject_outliers(data, z_thresh=3):
    #     if len(data) == 0:
    #         return np.array([])
    #     if np.std(data) == 0:
    #         return data
    #     z = np.abs((data - np.mean(data)) / np.std(data))
    #     return data[z < z_thresh]

    #     irAC_clean = reject_outliers(irAC)
    #     redAC_clean = reject_outliers(redAC)

    #     if len(irAC_clean) == 0 or len(redAC_clean) == 0 or irDC == 0 or redDC == 0:
    #         return np.nan

    #     # Mean of clean AC components
    #     irAC_mean = np.mean(irAC_clean)
    #     redAC_mean = np.mean(redAC_clean)

    #     ratio = (redAC_mean / redDC) / (irAC_mean / irDC)
    #     ratio = self.validRatioCheck(ratio)
    #     spo2 = 120 - 25 * ratio
    #     if plot:
    #         # Plot IR filtered signal with IR peaks
    #         t_ir = np.arange(len(irFilteredSignal)) / self.sf
    #         plt.figure(figsize=(10, 4))
    #         plt.plot(t_ir, irFilteredSignal, label="IR Signal", color="#33a02c")
    #         plt.plot(t_ir[ir_good_peaks], irFilteredSignal[ir_good_peaks], 'rx', label="IR Peaks", markersize=8)
    #         plt.title("IR filtered")
    #         plt.xlabel("Time (s)")
    #         plt.ylabel("Amplitude")
    #         plt.legend()
    #         plt.grid(True, linestyle="--", alpha=0.5)
    #         plt.tight_layout()
    #         plt.show()

    #         # Plot Red filtered signal with Red peaks
    #         t_red = np.arange(len(redFilteredSignal)) / self.sf
    #         plt.figure(figsize=(10, 4))
    #         plt.plot(t_red, redFilteredSignal, label="Red Signal", color="#ff7f0e")
    #         plt.plot(t_red[red_good_peaks], redFilteredSignal[red_good_peaks], 'rx', label="Red Peaks", markersize=8)
    #         plt.title("Red filtered")
    #         plt.xlabel("Time (s)")
    #         plt.ylabel("Amplitude")
    #         plt.legend()
    #         plt.grid(True, linestyle="--", alpha=0.5)
    #         plt.tight_layout()
    #         plt.show()

    #     return spo2, ratio
    def _reject_outliers(self, data, z_thresh=3.0):
        import numpy as np
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            return data
        s = np.std(data)
        if s == 0 or not np.isfinite(s):
            return data
        z = np.abs((data - np.mean(data)) / s)
        return data[z < z_thresh]

    def heartRateCalculation_NK(self, heartRateSignal, plot=False):
        try:
            signals, info = nk.ppg_process(heartRateSignal, sampling_rate=self.sf)
            heart_rate = signals["PPG_Rate"].mean()

            if plot:
                fig = nk.ppg_plot(signals, info)  # catch the figure
                plt.show(block=True)  # make sure it shows (optional: block=True to ensure it pauses if you want)
                plt.close(fig)

            return heart_rate
        except Exception as e:
            print(f"Error during heart rate calculation with NeuroKit2: {e}")
            return np.nan

    def analyze(self, greenPPG, irPPG, redPPG):
        green_heartRateSignal = self.bandpass_filter(greenPPG, self.greenCutOff)
        #heartRate = self.heartRateCalculation(green_heartRateSignal)
        heartRate = self.heartRateCalculation_NK(greenPPG)

        green_breathRateSignal = self.bandpass_filter(greenPPG, self.breathCutOff)
        breathingRate = self.breathingRateCalculation(green_breathRateSignal)


        irFilteredSignal = self.bandpass_filter(irPPG, self.irCutOff)
        redFilteredSignal = self.bandpass_filter(redPPG, self.redCutOff)

        # irFilteredSignal = self.fft_bandpass_filter(irPPG, self.sf, self.generalCutOff[0], self.generalCutOff[1])
        # redFilteredSignal = self.fft_bandpass_filter(redPPG, self.sf, self.generalCutOff[0], self.generalCutOff[1])

        irLow = self.lowpass_filter(irPPG, self.lowPassFreq)
        redLow = self.lowpass_filter(redPPG, self.lowPassFreq)

        spo2, ratio = self.calculateSP02_Separate(irLow, redLow, irFilteredSignal, redFilteredSignal, heartRate)

        return heartRate, breathingRate, spo2, ratio, green_heartRateSignal, green_breathRateSignal, irFilteredSignal, redFilteredSignal

    def compute_hrv_rmssd_sdnn(self, green_signal):
        """
        Compute HRV (RMSSD & SDNN in ms) using the green-light PPG signal.
        Expects a single window of green PPG already filtered in the heart-rate band.
        """
        x = np.asarray(green_signal, dtype=float)
        if x.size < int(0.8 * self.sf):
            return np.nan, np.nan

        # Peak detection on green PPG (higher SNR and stability)
        min_distance = int(0.3 * self.sf)  # ~300 ms refractory
        prom = max(np.std(x) * 0.25, 1e-12)
        peaks, _ = find_peaks(x, distance=min_distance, prominence=prom)

        if peaks.size < 2:
            return np.nan, np.nan

        # Inter-beat intervals (IBIs) in ms
        ibi_ms = np.diff(peaks) / self.sf * 1000.0
        if ibi_ms.size < 2:
            sdnn = np.std(ibi_ms, ddof=1) if ibi_ms.size > 1 else np.nan
            return np.nan, sdnn

        # HRV metrics
        sdnn = np.std(ibi_ms, ddof=1)
        diff_ibi = np.diff(ibi_ms)
        rmssd = np.sqrt(np.mean(diff_ibi ** 2)) if diff_ibi.size > 0 else np.nan

        return float(rmssd), float(sdnn)

    def compute_breath_amplitude(self, breathing_signal, plot=False):
        x = np.asarray(breathing_signal, dtype=float)
        if x.size < int(2 * self.sf):  # need at least ~2 s
            return np.nan, np.nan, np.nan, 0

        # Peaks (inspirations) and troughs (expirations)
        # Use same distance as your breathing rate detector (~2 s)
        min_distance = max(1, int(2.0 * self.sf))
        prom_peak = max(np.std(x) / 4.0, 1e-12)

        peaks, _   = find_peaks(x, distance=min_distance, prominence=prom_peak)
        troughs, _ = find_peaks(-x, distance=min_distance, prominence=prom_peak)

        if peaks.size == 0 or troughs.size == 0:
            return np.nan, np.nan, np.nan, 0

        # For each peak, find the last trough before it
        amps = []
        ti = 0  # trough index pointer
        for p in peaks:
            # advance ti to the last trough < p
            while ti + 1 < troughs.size and troughs[ti + 1] < p:
                ti += 1
            if ti < troughs.size and troughs[ti] < p:
                amp = x[p] - x[troughs[ti]]
                if np.isfinite(amp) and amp > 0:
                    amps.append(amp)

        if len(amps) == 0:
            return np.nan, np.nan, np.nan, 0

        amps = np.asarray(amps, dtype=float)
        median_amp = float(np.median(amps))
        mean_amp   = float(np.mean(amps))
        std_amp    = float(np.std(amps, ddof=1)) if amps.size > 1 else 0.0

        if plot:
            t = np.arange(len(x)) / float(self.sf)
            plt.figure(figsize=(10, 4))
            plt.plot(t, x, label="Breathing-band signal")
            plt.plot(t[peaks],   x[peaks],   "rx", label="Peaks")
            plt.plot(t[troughs], x[troughs], "kx", label="Troughs")
            plt.grid(True, ls="--", alpha=0.5)
            plt.legend(); plt.tight_layout(); plt.show()

        return median_amp, mean_amp, std_amp, int(amps.size)


