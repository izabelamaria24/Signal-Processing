import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'raw': 'cornflowerblue',
    'filtered': 'darkorange',
    'peaks': 'red',
    'dc_removed': 'seagreen'
}

def load_data(filename="Train.csv"):
    counts = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            counts.append(float(row[2]))
    return np.array(counts)


def solve_a_b_c(data, sampling_period_hours=1):
    # a)
    sampling_period_seconds = sampling_period_hours * 3600
    Fs = 1 / sampling_period_seconds
    print(f"(a) The data is sampled hourly. The sampling frequency (Fs) is 1 sample/hour, which is {Fs:.8f} Hz.")

    # b)
    num_samples = len(data)
    duration_hours = num_samples * sampling_period_hours
    duration_days = duration_hours / 24
    duration_years = duration_days / 365.25
    print(f"(b) The dataset contains {num_samples} hourly samples, covering a total of {duration_hours} hours, which is approximately {duration_days:.2f} days or {duration_years:.2f} years.")

    # c)
    nyquist_freq = Fs / 2
    print(f"(c) The maximum frequency present in the signal (Nyquist frequency) is Fs/2 = {nyquist_freq:.8f} Hz.")
    print("-" * 30 + "\n")


def solve_d(data, Fs):
    N = len(data)
    
    X = np.fft.fft(data)
    
    freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]
    
    X_mag = (2.0/N) * np.abs(X[:N//2])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, X_mag, color=COLORS['raw'])
    ax.set_title("Fourier Transform of the Traffic Signal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("exercise_d_fft.png")
    return freqs, X_mag


def solve_e(data, Fs):
    dc_component = np.mean(data)
    print(f"The signal has a DC component (non-zero mean) of {dc_component:.2f}.")
    
    data_no_dc = data - dc_component
    print("DC component removed by subtracting the mean from the signal.")
    
    N = len(data_no_dc)
    X = np.fft.fft(data_no_dc)
    freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]
    X_mag = (2.0/N) * np.abs(X[:N//2])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, X_mag, color=COLORS['dc_removed'])
    ax.set_title("Fourier Transform after DC Component Removal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("exercise_e_fft_no_dc.png")


def solve_f(freqs, X_mag):
    top_4_indices = np.argsort(X_mag)[::-1][:4]
    
    top_4_freqs = freqs[top_4_indices]
    top_4_mags = X_mag[top_4_indices]

    print("The 4 principal frequencies are:")
    for i in range(4):
        freq_hz = top_4_freqs[i]
        period_hours = 1 / (freq_hz * 3600)
        period_days = period_hours / 24
        
        print(f"  {i+1}. Freq: {freq_hz:.8f} Hz (Magnitude: {top_4_mags[i]:.2f}) -> Period: {period_hours:.2f} hours ({period_days:.2f} days)")

    print(" - The frequency around 1.157e-05 Hz corresponds to a 24-hour period (1 day), representing the daily traffic cycle.")
    print(" - The frequency around 2.315e-05 Hz corresponds to a 12-hour period (0.5 days), representing a semi-daily pattern (e.g., morning and evening rush hours).")
    print(" - The frequency around 1.653e-06 Hz corresponds to a 168-hour period (7 days), representing the weekly traffic cycle.")
    print(" - The other frequencies are likely harmonics of these fundamental frequencies.")
    print("-" * 30 + "\n")


def solve_g(data):
    start_day_index = 44
    start_sample_index = start_day_index * 24 # 1056
    
    num_days_to_plot = 30
    end_sample_index = start_sample_index + num_days_to_plot * 24
    
    data_slice = data[start_sample_index:end_sample_index]
    time_axis_days = np.arange(len(data_slice)) / 24.0
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time_axis_days, data_slice, label="Hourly Traffic Count", color=COLORS['raw'])
    
    for i in range(1, num_days_to_plot // 7 + 1):
        ax.axvline(x=i*7, color='gray', linestyle='--', linewidth=1)
        
    ax.set_title(f"One Month of Traffic Data (Starting from Day {start_day_index})")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Number of Cars")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("exercise_g_monthly_traffic.png")


def solve_h(data):
    print("1.  **Identify Weekly Pattern:** The signal has a strong weekly periodicity. We can determine the day of the week by averaging the traffic for each day over several weeks. The two consecutive days with the lowest average traffic would be Saturday and Sunday.")
    print("2.  **Major holidays, such as Christmas (December 25th), typically have a very distinct and sharp drop in traffic.")
    print("    - For example, we could look for the day with the minimum traffic in a window around where we expect Christmas to be (approximately 4 months after the start date of late August).")
    print("3.  **Cross-reference:** By combining the day of the week information from step 1 with the specific date anchor from step 2, we can determine the exact start date. For instance, if we identify December 25th and our weekly analysis shows it's a Tuesday, we can check the calendar for a year where this alignment occurs within our approximate timeframe.")
    
    print("\nPotential drawbacks:")
    print(" - **Atypical Events:** The method relies on typical traffic patterns. Anomalous events (e.g., major accidents, public events, pandemics) could distort the weekly or yearly averages.")
    print(" - **Holiday Ambiguity:** Some holidays have shifting dates (like Easter), which could be harder to pin down. A fixed-date holiday like Christmas is a more reliable anchor.")
    print(" - **Accuracy:** The accuracy depends on the stability of the traffic patterns. If the patterns change significantly over the two years, the averages might be misleading.")
    print("-" * 30 + "\n")


def solve_i(data, Fs):
    start_index = 1056
    data_slice = data[start_index : start_index + 30*24]
    time_axis_days = np.arange(len(data_slice)) / 24.0

    N = len(data_slice)
    X_slice = np.fft.fft(data_slice)
    freqs_slice = np.fft.fftfreq(N, 1/Fs)
    
    f_daily = 1 / (24 * 3600)
    cutoff_frequency = 10 * f_daily 
    
    X_filtered = X_slice.copy()
    X_filtered[np.abs(freqs_slice) > cutoff_frequency] = 0
    
    x_filtered = np.fft.ifft(X_filtered)
    print(f"Signal filtered by removing frequencies above {cutoff_frequency:.8f} Hz.")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time_axis_days, data_slice, label='Original Signal', color=COLORS['raw'], alpha=0.6)
    ax.plot(time_axis_days, x_filtered.real, label=f'Filtered Signal (cutoff={cutoff_frequency:.2e} Hz)', color=COLORS['filtered'], linewidth=2)
    ax.set_title("Original vs. Low-Pass Filtered Traffic Signal")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Number of Cars")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("exercise_i_filtered_signal.png")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Train.csv")
    traffic_data = load_data(file_path)
    
    FS = 1.0 / 3600.0 

    solve_a_b_c(traffic_data)
    
    freqs, X_mag = solve_d(traffic_data, FS)
    
    solve_e(traffic_data, FS)
    
    solve_f(freqs[1:], X_mag[1:]) 
    
    solve_g(traffic_data)
    
    solve_h(traffic_data)
    
    solve_i(traffic_data, FS)