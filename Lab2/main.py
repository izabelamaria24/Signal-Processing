import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from signal_processing_core import SignalToolkit, SignalType


tool = SignalToolkit()

SKIP_AUDIO = os.getenv('SKIP_AUDIO', '0').lower() in ('1', 'true', 'yes')


def ensure_lab_dir():
    return tool.lab_asset_path('Lab2')


def generate_signal_data(signal_type, freq, amplitude=1.0, phase=0.0, duration=None, num_samples=None, sampling_freq=8000):
    t, x = tool.generate_signal(signal_type, freq, amplitude=amplitude, phase=phase, sampling_freq=sampling_freq, duration=duration, num_samples=num_samples)
    return t, x


def play_audio_from_array(x, sampling_freq):
    if SKIP_AUDIO:
        print("SKIP_AUDIO enabled — skipping playback")
        return
    if np.max(np.abs(x)) == 0:
        print("Warning: audio array is silent")
        return
    scaled_data = np.int16(x / np.max(np.abs(x)) * 32767)
    sd.play(scaled_data, sampling_freq)
    sd.wait()


def save_wav_from_array(filename, x, sampling_freq):
    if np.max(np.abs(x)) == 0:
        scaled_data = np.zeros_like(x, dtype=np.int16)
    else:
        scaled_data = np.int16(x / np.max(np.abs(x)) * 32767)
    wavfile.write(filename, sampling_freq, scaled_data)
    print(f"Signal saved to {filename}")


def ex1():
    freq = 2
    amp = 1.5
    phase_sin = np.pi / 4
    duration = 2

    t1, x1 = generate_signal_data(SignalType.SINE, freq, amplitude=amp, phase=phase_sin, duration=duration)
    phase_cos = phase_sin - np.pi / 2
    t2, x2 = generate_signal_data(SignalType.COSINE, freq, amplitude=amp, phase=phase_cos, duration=duration)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Identical Sine and Cosine Signals", fontsize=16)

    ax1.plot(t1, x1)
    ax1.set_title("Sine")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    ax2.plot(t2, x2)
    ax2.set_title("Cosine")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    tool.save_figure(fig, 'ex1_sine_cosine', lab_name='Lab2')
    plt.close(fig)


def ex2():
    freq = 3
    amp = 1.0
    duration = 1.5
    phases = [0, np.pi / 4, np.pi / 2, np.pi]

    fig, ax = plt.subplots(figsize=(10, 6))
    for phase in phases:
        t, x = generate_signal_data(SignalType.SINE, freq, amplitude=amp, phase=phase, duration=duration)
        ax.plot(t, x, label=f"Phase = {phase/np.pi:.2f}π")

    ax.set_title(f"Sine Wave (f={freq}Hz) with Different Phases")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    tool.save_figure(fig, 'ex2a_phases', lab_name='Lab2')
    plt.close(fig)

    t_orig, x_orig = generate_signal_data(SignalType.SINE, freq, amplitude=amp, phase=phases[0], duration=duration)
    snr_values = [100, 10, 1, 0.1]

    fig, axes = plt.subplots(len(snr_values), 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Sine Wave with Added Noise at Different SNR", fontsize=16)

    for i, snr in enumerate(snr_values):
        z = np.random.normal(size=len(x_orig))
        norm_x = np.linalg.norm(x_orig)
        norm_z = np.linalg.norm(z)
        if norm_z == 0 or snr <= 0:
            noisy = x_orig
        else:
            gamma = (norm_x / norm_z) / np.sqrt(snr)
            noisy = x_orig + gamma * z

        axes[i].plot(t_orig, noisy)
        axes[i].set_title(f"Signal with SNR = {snr}")
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    tool.save_figure(fig, 'ex2b_snr', lab_name='Lab2')
    plt.close(fig)


def ex3():
    signals_to_play = [
        (SignalType.SINE, 400, None, 16000, 8000),
        (SignalType.SINE, 800, 3.0, None, 8000),
        (SignalType.SAWTOOTH, 240, 3.0, None, 8000),
        (SignalType.SQUARE, 300, 3.0, None, 8000),
    ]

    for stype, freq, duration, num_samples, fs in signals_to_play:
        if duration is not None:
            t, x = generate_signal_data(stype, freq, duration=duration, sampling_freq=fs)
        else:
            t, x = generate_signal_data(stype, freq, num_samples=num_samples, sampling_freq=fs)
        play_audio_from_array(x, fs)

    t0, x0 = generate_signal_data(SignalType.SINE, 400, num_samples=16000, sampling_freq=8000)
    wav_filename = 'sin_400hz.wav'
    save_wav_from_array(wav_filename, x0, 8000)
    rate, data = wavfile.read(wav_filename)
    print(f"Verification: Loaded {data.shape[0]} samples at a rate of {rate} Hz.")


def ex4():
    fs = 8000
    duration = 0.05
    t1, x1 = generate_signal_data(SignalType.SINE, 200, duration=duration, sampling_freq=fs)
    t2, x2 = generate_signal_data(SignalType.SAWTOOTH, 150, duration=duration, sampling_freq=fs)
    sum_data = x1 + x2

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Sum of Two Different Waveforms", fontsize=16)

    ax1.plot(t1, x1)
    ax1.set_title("Sine")
    ax1.grid(True)

    ax2.plot(t2, x2)
    ax2.set_title("Sawtooth")
    ax2.grid(True)

    ax3.plot(t1, sum_data)
    ax3.set_title("Sum of a Sine and a Sawtooth Signal")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    tool.save_figure(fig, 'ex4_sum', lab_name='Lab2')
    plt.close(fig)


def ex5():
    fs = 44100
    duration_per_note = 1.0

    t_low, x_low = generate_signal_data(SignalType.SINE, 440, duration=duration_per_note, sampling_freq=fs)
    t_high, x_high = generate_signal_data(SignalType.SINE, 880, duration=duration_per_note, sampling_freq=fs)

    concatenated_data = np.concatenate((x_low, x_high))
    play_audio_from_array(concatenated_data, fs)


def ex6():
    fs = 2000
    duration = 0.02

    freqs = {
        "f = fs/2 (Nyquist)": fs / 2,
        "f = fs/4": fs / 4,
        "f = 0 Hz": 0
    }

    fig, axes = plt.subplots(len(freqs), 1, figsize=(10, 9))
    fig.suptitle("Sine Waves at Special Frequencies (fs=2000Hz)", fontsize=16)

    for i, (title, freq) in enumerate(freqs.items()):
        t, x = generate_signal_data(SignalType.SINE, freq, duration=duration, sampling_freq=fs)
        axes[i].plot(t, x, '-o', markersize=4)
        axes[i].set_title(title)
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    tool.save_figure(fig, 'ex6_special_freqs', lab_name='Lab2')
    plt.close(fig)


def ex7():
    fs_initial = 1000
    freq = 50
    duration = 0.1

    t, x = generate_signal_data(SignalType.SINE, freq, duration=duration, sampling_freq=fs_initial)
    original_time = t

    decimated_data_1 = x[::4]
    decimated_time_1 = original_time[::4]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original_time, x, label='Original Signal', alpha=0.5)
    ax.plot(decimated_time_1, decimated_data_1, '-o', label='Decimated (start from 1st element)')
    ax.legend()
    ax.grid(True)
    tool.save_figure(fig, 'ex7a_decimation', lab_name='Lab2')
    plt.close(fig)

    decimated_data_2 = x[1::4]
    decimated_time_2 = original_time[1::4]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original_time, x, label='Original Signal', alpha=0.5)
    ax.plot(decimated_time_1, decimated_data_1, '-o', label='Decimated (start from 1st element)')
    ax.plot(decimated_time_2, decimated_data_2, '-s', label='Decimated (start from 2nd element)')
    ax.legend()
    ax.grid(True)
    tool.save_figure(fig, 'ex7b_decimation_phase', lab_name='Lab2')
    plt.close(fig)


def ex8():
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 500)

    y_true = np.sin(alpha)
    y_taylor = alpha
    y_pade = (alpha - (7 * alpha**3) / 60) / (1 + (alpha**2) / 20)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(alpha, y_true, label='sin(α) (True)')
    plt.plot(alpha, y_taylor, '--', label='Taylor Approx: α')
    plt.plot(alpha, y_pade, ':', label='Padé Approx')
    plt.title('Exercise 8: sin(α) and its Approximations')
    plt.xlabel('α (radians)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    tool.save_figure(fig, 'ex8a_approximations', lab_name='Lab2')
    plt.close(fig)

    error_taylor = np.abs(y_true - y_taylor)
    error_pade = np.abs(y_true - y_pade)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(alpha, error_taylor, label='|sin(α) - α| (Taylor Error)')
    ax1.plot(alpha, error_pade, label='|sin(α) - Padé| (Padé Error)')
    ax1.set_title('Approximation Error (Linear Scale)')
    ax1.set_xlabel('α (radians)')
    ax1.set_ylabel('Absolute Error')
    ax1.legend()
    ax1.grid(True)

    ax2.semilogy(alpha, error_taylor, label='|sin(α) - α| (Taylor Error)')
    ax2.semilogy(alpha, error_pade, label='|sin(α) - Padé| (Padé Error)')
    ax2.set_title('Approximation Error (Logarithmic Scale)')
    ax2.set_xlabel('α (radians)')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    tool.save_figure(fig, 'ex8b_errors', lab_name='Lab2')
    plt.close(fig)


if __name__ == '__main__':
    ensure_lab_dir()
    ex1()
    ex2()
    ex3()  
    ex4()
    ex5()  
    ex6()
    ex7()
    ex8()

