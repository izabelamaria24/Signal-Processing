import numpy as np
import matplotlib.pyplot as plt
from signal_processing_core import SignalToolkit

tool = SignalToolkit()

def construct_time_axis():
    return np.arange(0, 0.03, 0.00005)


def construct_signals_and_plot():
    t_x, x = tool.generate_signal(tool.SignalType.COSINE, freq=260, amplitude=1.0, phase=np.pi/3, sampling_freq=20000, duration=0.03)
    t_y, y = tool.generate_signal(tool.SignalType.COSINE, freq=140, amplitude=1.0, phase=-np.pi/3, sampling_freq=20000, duration=0.03)
    t_z, z = tool.generate_signal(tool.SignalType.COSINE, freq=60, amplitude=1.0, phase=np.pi/3, sampling_freq=20000, duration=0.03)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t_x, x)
    axes[0].set_title('x(t) = cos(520πt + π/3)')
    axes[1].plot(t_y, y)
    axes[1].set_title('y(t) = cos(280πt − π/3)')
    axes[2].plot(t_z, z)
    axes[2].set_title('z(t) = cos(120πt + π/3)')

    for ax in axes:
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

    plt.tight_layout()
    tool.save_figure(fig, 'signals_ex1', lab_name='Lab1')
    plt.show()


def sample_time_series():
    t, x = tool.generate_signal(tool.SignalType.COSINE, freq=260, amplitude=1.0, phase=np.pi/3, sampling_freq=200, duration=0.03)
    _, y = tool.generate_signal(tool.SignalType.COSINE, freq=140, amplitude=1.0, phase=-np.pi/3, sampling_freq=200, duration=0.03)
    _, z = tool.generate_signal(tool.SignalType.COSINE, freq=60, amplitude=1.0, phase=np.pi/3, sampling_freq=200, duration=0.03)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].stem(t, x)
    axes[0].set_title('x[n] = cos(520πnTs + π/3)')
    axes[1].stem(t, y)
    axes[1].set_title('y[n] = cos(280πnTs − π/3)')
    axes[2].stem(t, z)
    axes[2].set_title('z[n] = cos(120πnTs + π/3)')

    for ax in axes:
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

    plt.tight_layout()
    tool.save_figure(fig, 'sampled_signals_ex1', lab_name='Lab1')
    plt.show()


if __name__ == '__main__':
    construct_signals_and_plot()
    sample_time_series()
