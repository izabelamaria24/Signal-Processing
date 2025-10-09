# semnalele continue x(t) = cos(520πt + π/3), y(t) = cos(280πt − π/3), z(t) = cos(120πt + π/3)

import numpy as np
import matplotlib.pyplot as plt

# a)
def construct_time_axis():
   time = np.arange(0, 0.03, 0.00005)
   return time

# b)
def construct_signals_and_plot():
    t = construct_time_axis()

    x = np.cos(520 * np.pi * t + np.pi / 3)
    y = np.cos(280 * np.pi * t - np.pi / 3)
    z = np.cos(120 * np.pi * t + np.pi / 3)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title('x(t) = cos(520πt + π/3)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.title('y(t) = cos(280πt − π/3)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, z)
    plt.title('z(t) = cos(120πt + π/3)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("signals_ex1.pdf", format="eps")
    plt.show()


def sample_time_series():
    frequency = 200 # hz
    ts = 1 / frequency
    t = np.arange(0, 0.03, ts)

    x = np.cos(520 * np.pi * t + np.pi / 3)
    y = np.cos(280 * np.pi * t - np.pi / 3)
    z = np.cos(120 * np.pi * t + np.pi / 3)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.stem(t, x)
    plt.title('x(t) = cos(520πnTs + π/3)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.stem(t, y)
    plt.title('y(t) = cos(280πnTs − π/3)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.stem(t, z)
    plt.title('z(t) = cos(120πnTs + π/3)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("sampled_signals_ex1.pdf", format="eps")
    plt.show()

construct_signals_and_plot()
sample_time_series()
