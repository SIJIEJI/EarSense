import numpy as np
import pandas as pd


def calculate_snr_from_psd(psd_values, freqs, target_frequency, band_range=5, target_band=0.2):
    """
    Calculate the SNR of a target frequency range compared to the surrounding frequency range in the PSD data for a single channel.

    Parameters:
        psd_values (numpy.ndarray): PSD values for a single EEG channel.
        freqs (numpy.ndarray): Array of corresponding frequency values for the PSD.
        target_frequency (float): The center of the frequency range at which to calculate SNR.
        band_range (float): The range around the target frequency for noise estimation, in Hz (default is ±5 Hz).
        target_band (float): The half-width of the target frequency range, in Hz (default is ±0.2 Hz).

    Returns:
        snr (float): The signal-to-noise ratio at the target frequency range.
    """
    # Define indices for the target frequency range
    target_indices = np.where(
        (freqs >= target_frequency - target_band) &
        (freqs <= target_frequency + target_band)
    )[0]

    # Calculate signal power as the average PSD within the target frequency range
    signal_power = np.mean(psd_values[target_indices])

    # Define indices for the surrounding noise frequency range, excluding the target frequency range
    noise_indices = np.where(
        (freqs >= target_frequency - band_range) &
        (freqs <= target_frequency + band_range) &
        ~np.isin(np.arange(len(freqs)), target_indices)  # Exclude target frequency indices
    )[0]

    # Calculate the noise power as the average PSD in the surrounding frequency range
    noise_power = np.mean(psd_values[noise_indices])

    # Calculate SNR as the ratio of the signal power to the noise power
    snr = signal_power / noise_power

    return snr


def load_psd_data(file_path):
    """
    Load PSD data from a tab-separated .txt file with a header and multiple columns (frequency and PSD channels).

    Parameters:
        file_path (str): Path to the PSD data file.

    Returns:
        freqs (numpy.ndarray): Frequency values.
        psd_channels (list of numpy.ndarray): List of PSD values for each channel.
    """
    # Load the data file, skipping the first row and using tab as the delimiter
    # Specify encoding to handle non-UTF-8 files
    data = pd.read_csv(file_path, skiprows=1, delimiter='\t', encoding='ISO-8859-1')
    freqs = data.iloc[:, 0].to_numpy()  # First column as frequency array
    psd_channels = [data.iloc[:, i].to_numpy() for i in range(1, data.shape[1])]  # Remaining columns as PSD data

    return freqs, psd_channels


# Example usage
if __name__ == "__main__":
    # Example file path (replace with actual path to the PSD data file)
    file_path = r"xxx.txt"

    # Load PSD data from file
    freqs, psd_channels = load_psd_data(file_path)

    # Define target frequency, surrounding band range for noise, and target frequency band for signal calculation
    target_frequency = 10  # Target frequency in Hz
    band_range = 5  # Frequency range around the target for noise calculation
    target_band = 2  # Frequency range around the target for signal calculation

    # Calculate SNR for each channel
    snr_values = []
    for i, psd_values in enumerate(psd_channels):
        snr = calculate_snr_from_psd(psd_values, freqs, target_frequency, band_range, target_band)
        snr_values.append(snr)
        print(f" {snr:.2f}")


