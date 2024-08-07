import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET


def parse_stream_file(file_path):
    """
    Parse the ECG data and metadata from the XML stream file.

    Parameters:
    file_path (str): Path to the XML stream file.

    Returns:
    ecg_signal (numpy array): The extracted ECG signal.
    sampling_rate (float): Sampling rate of the ECG data.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract sampling rate from the info element
    info_element = root.find('info')
    if info_element is not None:
        # Default to 1.0 if sr attribute not found
        sampling_rate = float(info_element.get('sr', 1.0))
    else:
        raise ValueError("Info element not found in the XML structure.")

    # Extract ECG signal from the chunk element
    chunk_element = root.find('chunk')
    if chunk_element is not None:
        num_value = int(chunk_element.get('num', 0))
        if num_value > 0:
            # Replace with actual data extraction if available
            ecg_signal = np.random.rand(num_value)
            return ecg_signal, sampling_rate
        else:
            raise ValueError("Invalid 'num' attribute in chunk element.")
    else:
        raise ValueError("Chunk element not found in the XML structure.")


def compute_ecg_duration(ecg_signal, sampling_rate):
    """
    Compute the duration of the ECG signal.

    Parameters:
    ecg_signal (numpy array): The ECG signal array.
    sampling_rate (float): The sampling rate of the ECG signal in Hz.

    Returns:
    duration (float): The duration of the ECG signal in seconds.
    """
    number_of_samples = len(ecg_signal)
    duration = number_of_samples / sampling_rate
    return duration


def compute_heart_rate_over_time(ecg_signal, sampling_rate, window_size, step_size):
    """
    Compute the heart rate over time from ECG signal using a sliding window approach.

    Parameters:
    ecg_signal (numpy array): The ECG signal array.
    sampling_rate (int): The sampling rate of the ECG signal in Hz.
    window_size (float): The size of the sliding window in seconds.
    step_size (float): The step size for the sliding window in seconds.

    Returns:
    hr_over_time (list): The computed heart rates over time in beats per minute (bpm).
    time_stamps (list): The time stamps corresponding to each computed heart rate.
    """
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    hr_over_time = []
    time_stamps = []

    for start in range(0, len(ecg_signal) - window_samples, step_samples):
        end = start + window_samples
        window_signal = ecg_signal[start:end]

        # Adjusted distance parameter
        distance = max(int(sampling_rate / 2.5), 1)

        peaks, _ = find_peaks(window_signal, distance=distance, height=np.mean(window_signal))

        if len(peaks) > 1:  # Ensure there are at least two peaks to compute RR intervals
            rr_intervals = np.diff(peaks) / sampling_rate  # Convert sample intervals to time intervals in seconds
            heart_rates = 60 / rr_intervals  # Convert RR intervals to heart rates in bpm
        else:
            rr_intervals = []
            heart_rates = np.nan  # If there are not enough peaks, assign NaN

        if isinstance(heart_rates, np.ndarray):  # Check if heart_rates is an array
            if len(heart_rates) > 0:
                hr_over_time.append(np.nanmean(heart_rates))  # Average heart rate within the window
            else:
                hr_over_time.append(np.nan)
        else:
            hr_over_time.append(np.nan)  # Append NaN if heart_rates is not an array

        time_stamps.append(start / sampling_rate + window_size / 2)

    return hr_over_time, time_stamps


def plot_heart_rate_over_time(ecg_signal, hr_over_time, time_stamps, sampling_rate, window_size):
    """
    Plot the heart rate over time and the ECG signal with detected peaks for the first window.

    Parameters:
    ecg_signal (numpy array): The ECG signal array.
    hr_over_time (list): The computed heart rates over time in beats per minute (bpm).
    time_stamps (list): The time stamps corresponding to each computed heart rate.
    sampling_rate (float): The sampling rate of the ECG signal in Hz.
    window_size (float): The size of the sliding window in seconds.
    """
    print("Time Stamps:", time_stamps)

    # Plot heart rate over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps[:len(hr_over_time)], hr_over_time, label='Heart Rate (bpm)', marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (bpm)')
    plt.title('Heart Rate Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot ECG signal with detected peaks (for the first window as an example)
    plt.figure(figsize=(10, 6))
    peaks, _ = find_peaks(ecg_signal[:int(window_size * sampling_rate)], distance=max(1, int(sampling_rate / 2.5)),
                          height=np.mean(ecg_signal[:int(window_size * sampling_rate)]))
    plt.plot(np.arange(len(ecg_signal[:int(window_size * sampling_rate)])) / sampling_rate,
             ecg_signal[:int(window_size * sampling_rate)], label='ECG Signal')
    plt.plot(peaks / sampling_rate, ecg_signal[peaks], 'ro', label='Detected Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Detected Peaks (First Window)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    xml_file_path = 'jacket-study/SSJ/2024-07-07-12-23-00/ecg.stream'

    try:
        # Parse ECG data and metadata from XML file
        ecg_signal, sampling_rate = parse_stream_file(xml_file_path)
        duration = compute_ecg_duration(ecg_signal, sampling_rate)
        print(f"ECG Duration: {duration:.2f} seconds")
        print(f"ECG Signal Length: {len(ecg_signal)}")
        print(f"ECG Signal: {ecg_signal}")
        print(f"Sampling Rate: {sampling_rate}")
        # Generate time axis based on the sampling rate
        sampling_rate = 1.0
        duration = len(ecg_signal)
        time = np.arange(0, duration, 1 / sampling_rate)

        # Plot the ECG signal
        plt.figure(figsize=(12, 4))
        plt.plot(time, ecg_signal, label='ECG Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('ECG Signal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Compute heart rate over time
        window_size = 5  # 5-second window
        step_size = 1  # 1-second step
        hr_over_time, time_stamps = compute_heart_rate_over_time(ecg_signal, sampling_rate, window_size, step_size)

        # Plot heart rate over time and ECG signal with detected peaks
        plot_heart_rate_over_time(ecg_signal, hr_over_time, time_stamps, sampling_rate, window_size)

        # Save computed heart rates to CSV file
        hr_df = pd.DataFrame({
            'Time (s)': time_stamps,
            'Heart Rate (bpm)': hr_over_time
        })
        hr_df.to_csv('heart_rate.csv', index=False)

        # Interpolate heart rates over the entire duration
        interpolator = interp1d(time_stamps, hr_over_time, kind='linear', bounds_error=False, fill_value="extrapolate")
        hr_interpolated = interpolator(np.linspace(0, duration, len(ecg_signal)))

        # Save interpolated heart rates to CSV file
        hr_interpolated_df = pd.DataFrame({
            'Time (s)': np.linspace(0, duration, len(ecg_signal)),
            'Heart Rate (bpm)': hr_interpolated
        })
        hr_interpolated_df.to_csv('heart_rate_interpolated.csv', index=False)

    except Exception as e:
        print(f"Errorr: {e}")