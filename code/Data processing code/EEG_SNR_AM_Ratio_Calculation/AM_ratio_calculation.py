import numpy as np
import pandas as pd


def calculate_average_psd_in_range(freqs, psd_values, range_min, range_max):
    """
    Calculate the average PSD over a specified frequency range for a single channel.

    Parameters:
        freqs (numpy.ndarray): Array of frequency values.
        psd_values (numpy.ndarray): PSD values corresponding to each frequency for a single channel.
        range_min (float): Minimum frequency of the range.
        range_max (float): Maximum frequency of the range.

    Returns:
        average_psd (float): The average PSD within the specified frequency range.
    """
    # Find indices for the frequency range
    range_indices = np.where((freqs >= range_min) & (freqs <= range_max))[0]

    # Check if any indices were found in the specified range
    if range_indices.size > 0:
        psd_in_range = psd_values[range_indices]

        average_psd = np.mean(psd_in_range)
    else:

        average_psd = 0  # Return 0 or handle as needed

    return average_psd


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
    data = pd.read_csv(file_path, skiprows=1, delimiter='\t', encoding='ISO-8859-1')
    freqs = data.iloc[:, 0].to_numpy(dtype=np.float64)  # First column as frequency array
    psd_channels = [data.iloc[:, i].to_numpy(dtype=np.float64) for i in
                    range(1, data.shape[1])]  # Remaining columns as PSD data

    return freqs, psd_channels


# Example usage
if __name__ == "__main__":
    # Example file path (replace with actual path to the PSD data file)
    file_path = r"xxx.txt"

    # Load PSD data from file
    freqs, psd_channels = load_psd_data(file_path)

    # Define the frequency range for averaging
    range_min = 8  # Minimum frequency of the range
    range_max = 12  # Maximum frequency of the range

    # Calculate the average PSD in the specified range for each channel
    for i, psd_values in enumerate(psd_channels):
        average_psd = calculate_average_psd_in_range(freqs, psd_values, range_min, range_max)
        print(f"{average_psd:.10e}")

