import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import sounddevice as sd
from scipy.io import wavfile
import os

if not os.path.exists('Lab2/charts'):
    os.makedirs('Lab2/charts')

class SignalType(Enum):
    SINE = 0
    SAWTOOTH = 1
    SQUARE = 2
    COSINE = 3

class Signal:
    def __init__(self, name, freq, signal_type, amplitude=1.0, phase=0.0, duration=None, num_samples=None, sampling_freq=8000):
        self.name = name
        self.freq = freq
        self.signal_type = signal_type
        self.amplitude = amplitude
        self.phase = phase
        self.sampling_freq = sampling_freq

        if duration is not None:
            self.duration = duration
            self.time_series = np.arange(0, self.duration, 1 / self.sampling_freq)
        elif num_samples is not None:
            self.num_samples = num_samples
            n = np.arange(num_samples)
            self.time_series = n / self.sampling_freq
        else:
            raise ValueError("Please provide either duration or num_samples.")

    def generate_signal_data(self):
        omega = 2 * np.pi * self.freq
        if self.signal_type == SignalType.SINE:
            return self.amplitude * np.sin(omega * self.time_series + self.phase)
        elif self.signal_type == SignalType.SAWTOOTH:
            t = self.time_series
            f = self.freq
            return self.amplitude * 2 * (t * f - np.floor(0.5 + t * f))
        elif self.signal_type == SignalType.SQUARE:
            return self.amplitude * np.sign(np.sin(omega * self.time_series + self.phase))
        elif self.signal_type == SignalType.COSINE:
            return self.amplitude * np.cos(omega * self.time_series + self.phase)
        return 0

    def add_noise(self, snr):
        x = self.generate_signal_data()
        z = np.random.normal(size=len(x))
        
        norm_x = np.linalg.norm(x) 
        norm_z = np.linalg.norm(z) 
        
        if norm_z == 0 or snr < 0: return x 
        
        gamma = (norm_x / norm_z) / np.sqrt(snr)
        
        return x + gamma * z

    def play_audio(self):
        signal_data = self.generate_signal_data()
        scaled_data = np.int16(signal_data / np.max(np.abs(signal_data)) * 32767)
        print(f"Playing audio for: {self.name}...")
        sd.play(scaled_data, self.sampling_freq) 
        sd.wait()
        print("Playback finished.")

    def save_wav(self, filename):
        signal_data = self.generate_signal_data()
        scaled_data = np.int16(signal_data / np.max(np.abs(signal_data)) * 32767)
        wavfile.write(filename, self.sampling_freq, scaled_data) 
        print(f"Signal saved to {filename}")

    @staticmethod
    def load_wav(filename):
        rate, data = wavfile.read(filename) 
        print(f"Signal loaded from {filename} with sample rate {rate}")
        return rate, data

    def plot(self, ax, data=None, title_override=None):
        signal_data = data if data is not None else self.generate_signal_data()
        ax.plot(self.time_series, signal_data)
        ax.set_title(title_override if title_override else self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        
        
def ex1():
    freq = 2
    amp = 1.5
    phase_sin = np.pi / 4
    duration = 2

    s_sin = Signal(
        "Sine Signal (f=2Hz, A=1.5, φ=π/4)",
        freq, SignalType.SINE, amplitude=amp, phase=phase_sin, duration=duration
    )

    phase_cos = phase_sin - np.pi / 2
    s_cos = Signal(
        f"Cosine Signal",
        freq, SignalType.COSINE, amplitude=amp, phase=phase_cos, duration=duration
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Identical Sine and Cosine Signals", fontsize=16)

    s_sin.plot(ax1)
    s_cos.plot(ax2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Lab2/charts/ex1_sine_cosine.pdf')
    plt.show()

ex1()


def ex2():
    freq = 3
    amp = 1.0 
    duration = 1.5
    phases = [0, np.pi / 4, np.pi / 2, np.pi]

    fig, ax = plt.subplots(figsize=(10, 6))
    for phase in phases:
        s = Signal(
            f"Sine (φ={phase/np.pi:.2f}π)",
            freq, SignalType.SINE, amplitude=amp, phase=phase, duration=duration
        )
        data = s.generate_signal_data()
        ax.plot(s.time_series, data, label=f"Phase = {phase/np.pi:.2f}π")

    ax.set_title(f"Sine Wave (f={freq}Hz) with Different Phases")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    plt.savefig('Lab2/charts/ex2a_phases.pdf')
    plt.show()

    signal_to_corrupt = Signal(
        "Original Sine Wave", freq, SignalType.SINE, amplitude=amp, phase=phases[0], duration=duration
    )
    snr_values = [100, 10, 1, 0.1]
    
    fig, axes = plt.subplots(len(snr_values), 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Sine Wave with Added Noise at Different SNR", fontsize=16)

    for i, snr in enumerate(snr_values):
        noisy_data = signal_to_corrupt.add_noise(snr)
        signal_to_corrupt.plot(axes[i], data=noisy_data, title_override=f"Signal with SNR = {snr}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('Lab2/charts/ex2b_snr.pdf')
    plt.show()


ex2()


def ex3():
    signals_to_play = [
        Signal("a) Sin 400 Hz", 400, SignalType.SINE, num_samples=16000, sampling_freq=8000),
        Signal("b) Sin 800 Hz", 800, SignalType.SINE, duration=3, sampling_freq=8000),
        Signal("c) Sawtooth 240 Hz", 240, SignalType.SAWTOOTH, duration=3, sampling_freq=8000),
        Signal("d) Square 300 Hz", 300, SignalType.SQUARE, duration=3, sampling_freq=8000)
    ]

    for s in signals_to_play:
        s.play_audio()

    wav_filename = 'sin_400hz.wav'
    signals_to_play[0].save_wav(wav_filename)

    rate, data = Signal.load_wav(wav_filename)
    print(f"Verification: Loaded {data.shape[0]} samples at a rate of {rate} Hz.")


ex3()


def ex4():
    fs = 8000
    duration = 0.05
    
    s1 = Signal("Sinusoidal (f=200Hz)", 200, SignalType.SINE, duration=duration, sampling_freq=fs)
    s2 = Signal("Sawtooth (f=150Hz)", 150, SignalType.SAWTOOTH, duration=duration, sampling_freq=fs)

    data1 = s1.generate_signal_data()
    data2 = s2.generate_signal_data()
    sum_data = data1 + data2

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Sum of Two Different Waveforms", fontsize=16)

    s1.plot(ax1, data=data1)
    s2.plot(ax2, data=data2)
    
    ax3.plot(s1.time_series, sum_data)
    ax3.set_title("Sum of a Sine and a Sawtooth Signal")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Lab2/charts/ex4_sum.pdf')
    plt.show()


ex4()


def ex5():
    fs = 44100
    duration_per_note = 1.0

    s_low = Signal("Sine (f=440Hz, Note A4)", 440, SignalType.SINE, duration=duration_per_note, sampling_freq=fs)
    s_high = Signal("Sine (f=880Hz, Note A5)", 880, SignalType.SINE, duration=duration_per_note, sampling_freq=fs)

    data_low = s_low.generate_signal_data()
    data_high = s_high.generate_signal_data()

    concatenated_data = np.concatenate((data_low, data_high))
    
    scaled_data = np.int16(concatenated_data / np.max(np.abs(concatenated_data)) * 32767)
    sd.play(scaled_data, fs)
    sd.wait()


ex5()


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
        s = Signal(title, freq, SignalType.SINE, amplitude=1, phase=0, duration=duration, sampling_freq=fs)
        data = s.generate_signal_data()
        axes[i].plot(s.time_series, data, '-o', markersize=4)
        axes[i].set_title(title)
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Lab2/charts/ex6_special_freqs.pdf')
    plt.show()


ex6()


def ex7():
    fs_initial = 1000
    freq = 50 
    duration = 0.1

    s = Signal(f"Original Signal (fs={fs_initial}Hz)", freq, SignalType.SINE, duration=duration, sampling_freq=fs_initial)
    original_data = s.generate_signal_data()
    original_time = s.time_series

    decimated_data_1 = original_data[::4]
    decimated_time_1 = original_time[::4]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original_time, original_data, label='Original Signal', alpha=0.5)
    ax.plot(decimated_time_1, decimated_data_1, '-o', label='Decimated (start from 1st element)')
    ax.legend()
    ax.grid(True)
    plt.savefig('Lab2/charts/ex7a_decimation.pdf')
    plt.show()

    decimated_data_2 = original_data[1::4]
    decimated_time_2 = original_time[1::4]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original_time, original_data, label='Original Signal', alpha=0.5)
    ax.plot(decimated_time_1, decimated_data_1, '-o', label='Decimated (start from 1st element)')
    ax.plot(decimated_time_2, decimated_data_2, '-s', label='Decimated (start from 2nd element)')
    ax.legend()
    ax.grid(True)
    plt.savefig('Lab2/charts/ex7b_decimation_phase.pdf')
    plt.show()


ex7()


def ex8():
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 500)
    
    y_true = np.sin(alpha)
    y_taylor = alpha
    y_pade = (alpha - (7 * alpha**3) / 60) / (1 + (alpha**2) / 20)

    plt.figure(figsize=(10, 6))
    plt.plot(alpha, y_true, label='sin(α) (True)')
    plt.plot(alpha, y_taylor, '--', label='Taylor Approx: α')
    plt.plot(alpha, y_pade, ':', label='Padé Approx')
    plt.title('Exercise 8: sin(α) and its Approximations')
    plt.xlabel('α (radians)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('Lab2/charts/ex8a_approximations.pdf')
    plt.show()

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
    plt.savefig('Lab2/charts/ex8b_errors.pdf')
    plt.show()

ex8()

