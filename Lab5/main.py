import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
try:
    import pandas as pd 
except Exception:
    pd = None

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab5'
tool = SignalToolkit()

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

def load_dataframe(filename="Train.csv"):
    if pd is not None:
        df = pd.read_csv(filename)
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
        return df[['Datetime', 'Count']]

    datetimes = []
    counts = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            dt = datetime.strptime(row[1], "%d-%m-%Y %H:%M")
            datetimes.append(dt)
            counts.append(float(row[2]))
    return (np.array(datetimes, dtype='datetime64[m]'), np.array(counts, dtype=float))


def solve_a_b_c(data, sampling_period_hours=1):
    sampling_period_seconds = sampling_period_hours * 3600
    Fs = 1 / sampling_period_seconds
    print(f"(a) The data is sampled hourly. The sampling frequency (Fs) is 1 sample/hour, which is {Fs:.8f} Hz.")

    num_samples = len(data)
    duration_hours = num_samples * sampling_period_hours
    duration_days = duration_hours / 24
    duration_years = duration_days / 365.25
    print(f"(b) The dataset contains {num_samples} hourly samples, covering a total of {duration_hours} hours, which is approximately {duration_days:.2f} days or {duration_years:.2f} years.")

    nyquist_freq = Fs / 2
    print(f"(c) The maximum frequency present in the signal (Nyquist frequency) is Fs/2 = {nyquist_freq:.8f} Hz.")
    print("-" * 30 + "\n")


def solve_d(data, Fs):
    N = len(data)
    freqs = Fs * np.linspace(0, N // 2, N // 2) / N
    X = np.abs(np.fft.fft(data) / N)[: N // 2]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, X, color=COLORS['raw'])
    ax.set_title("FFT with DC component")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_yscale("log")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    tool.save_figure(fig, "exercise_d_fft", lab_name=LAB_NAME)
    plt.show()
    return freqs, X


def solve_e(data, Fs):
    N = len(data)
    freqs = Fs * np.linspace(0, N // 2, N // 2) / N
    X = np.abs(np.fft.fft(data) / N)[: N // 2].copy()
    dc_component = X[0]
    print(f"The signal has a DC component (non-zero mean) of ~{dc_component:.2f} (FFT bin at 0 Hz).")
    X[0] = 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, X, color=COLORS['dc_removed'])
    ax.set_title("FFT without the DC component")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_yscale("log")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    tool.save_figure(fig, "exercise_e_fft_no_dc", lab_name=LAB_NAME)
    plt.show()


def solve_f(freqs, X_mag):
    top_4_indices = np.argsort(X_mag)[::-1][:4]
    top_4_freqs = freqs[top_4_indices]
    top_4_mags = X_mag[top_4_indices]

    print("The 4 principal frequencies are:")
    for i in range(4):
        freq_hz = top_4_freqs[i]
        period_hours = 1 / (freq_hz * 3600) if freq_hz != 0 else float('inf')
        period_days = period_hours / 24 if period_hours != float('inf') else float('inf')
        print(f"  {i+1}. Freq: {freq_hz:.8f} Hz (Magnitude: {top_4_mags[i]:.2f}) -> Period: {period_hours:.2f} hours ({period_days:.2f} days)")

    print(" - The frequency around 1.157e-05 Hz corresponds to a 24-hour period (1 day), representing the daily traffic cycle.")
    print(" - The frequency around 2.315e-05 Hz corresponds to a 12-hour period (0.5 days), representing a semi-daily pattern (e.g., morning and evening rush hours).")
    print(" - The frequency around 1.653e-06 Hz corresponds to a 168-hour period (7 days), representing the weekly traffic cycle.")
    print(" - The other frequencies are likely harmonics of these fundamental frequencies.")
    print("-" * 30 + "\n")


def solve_g(df_or_tuple):
    start_dt = datetime(2013, 4, 1, 0, 0, 0)
    end_dt = datetime(2013, 5, 1, 0, 0, 0)

    if pd is not None and not isinstance(df_or_tuple, tuple):
        df = df_or_tuple
        mask = (df['Datetime'] >= start_dt) & (df['Datetime'] < end_dt)
        series = df.loc[mask, 'Count'].to_numpy()
    else:
        datetimes_np, counts_np = df_or_tuple
        dtn = np.array([np.datetime64(start_dt), np.datetime64(end_dt)])
        mask = (datetimes_np >= dtn[0]) & (datetimes_np < dtn[1])
        series = counts_np[mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series, label="Hourly Traffic Count", color=COLORS['raw'])
    ax.set_title("April 2013 — Hourly Traffic")
    ax.set_xlabel("Hour index within April")
    ax.set_ylabel("Number of Cars")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    tool.save_figure(fig, "exercise_g_april_2013", lab_name=LAB_NAME)
    plt.show()


def solve_h(data):
    pass


def solve_i(data, Fs):
    N = len(data)
    freqs = Fs * np.linspace(0, N // 2, N // 2) / N
    X = np.abs(np.fft.fft(data) / N)[: N // 2].copy()
    X[0] = 0.0 

    cutoff = 1.0 / (7 * 24 * 3600)
    mask = freqs < cutoff

    fig, ax = plt.subplots(figsize=(12, 6))
    container = ax.stem(freqs[mask], X[mask])
    try:
        container.markerline.set_markerfacecolor('none')
        baseline = container.baseline
    except Exception:
        try:
            markerline, stemlines, baseline = container
            markerline.set_markerfacecolor('none')
        except Exception:
            baseline = None
    ax.set_title("Low-frequency spectrum (periods ≥ 1 week)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    if baseline is not None:
        plt.setp(baseline, linewidth=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    tool.save_figure(fig, "exercise_i_low_freq_spectrum", lab_name=LAB_NAME)
    plt.show()


def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Train.csv")
    df_or_tuple = load_dataframe(file_path)
    if pd is not None and not isinstance(df_or_tuple, tuple):
        traffic_data = df_or_tuple['Count'].to_numpy()
    else:
        _, counts_np = df_or_tuple
        traffic_data = counts_np

    FS = 1.0 / 3600.0
    solve_a_b_c(traffic_data)
    freqs, X_mag = solve_d(traffic_data, FS)
    solve_e(traffic_data, FS)
    solve_f(freqs[1:], X_mag[1:])
    solve_g(df_or_tuple)
    solve_h(traffic_data)
    solve_i(traffic_data, FS)


if __name__ == "__main__":
    run()