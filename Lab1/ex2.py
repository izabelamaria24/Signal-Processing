import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class SignalType(Enum):
    SINE = 0
    SAWTOOTH = 1
    SQUARE = 2


class Signal:
    def __init__(self, name, freq, signal_type, duration=None, num_samples=None, sampling_freq=8000):
        self.name = name
        self.freq = freq
        self.signal_type = signal_type
        self.sampling_freq = sampling_freq

        if duration is not None:
            self.duration = duration
            self.time_series = np.arange(0, self.duration, 1 / self.sampling_freq)
        elif num_samples is not None:
            self.num_samples = num_samples
            n = np.arange(self.num_samples)
            self.time_series = n / self.sampling_freq
        else:
            raise ValueError("Please provide either duration or num_samples.")

    def generate_signal_data(self):
        if self.signal_type == SignalType.SINE:
            # Formula for a sine wave
            return np.sin(2 * np.pi * self.freq * self.time_series)
        elif self.signal_type == SignalType.SAWTOOTH:
            # Formula for a sawtooth wave
            t = self.time_series
            f = self.freq
            return 2 * (t * f - np.floor(0.5 + t * f))
        elif self.signal_type == SignalType.SQUARE:
            # Formula for a square wave
            return np.sign(np.sin(2 * np.pi * self.freq * self.time_series))
        return 0

    def plot(self, ax):
        signal_data = self.generate_signal_data()

        ax.plot(self.time_series, signal_data)
        ax.set_title(self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True)


def plot_and_save_2d(data, title, filename, colormap):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap=colormap)
    plt.title(title)
    plt.colorbar()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


signals_to_plot = [
    {"signal": Signal("a) Sin 400 Hz with 1600 samples", 400, SignalType.SINE, num_samples=1600),
     "filename": "charts/a_sinus_400Hz.pdf"},
    {"signal": Signal("b) Sin 800 Hz, duration 3s", 800, SignalType.SINE, duration=3), "filename": "charts/b_sinus_800Hz.pdf"},
    {"signal": Signal("c) Sawtooth 240 Hz", 240, SignalType.SAWTOOTH, duration=0.02),
     "filename": "charts/c_sawtooth_240Hz.pdf"},
    {"signal": Signal("d) Square 300 Hz", 300, SignalType.SQUARE, duration=0.01), "filename": "charts/d_square_300Hz.pdf"}
]

for item in signals_to_plot:
    s = item["signal"]
    filename = item["filename"]

    fig, ax = plt.subplots(figsize=(10, 6))

    s.plot(ax)

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

# e) 128x128 Random 2D Signal
random_signal_2d = np.random.rand(128, 128)
plot_and_save_2d(random_signal_2d, 'e) Random 2D Signal (128x128)', 'charts/e_random_2D.pdf', 'gray')

# f) Custom 128x128 2D Signal (Horizontal Gradient)
custom_signal_2d = np.zeros((128, 128))
gradient = np.linspace(0, 1, 128)
custom_signal_2d[:, :] = gradient
plot_and_save_2d(custom_signal_2d, 'f) Custom 2D Signal (Horizontal Gradient)', 'charts/f_custom_2D.pdf', 'plasma')