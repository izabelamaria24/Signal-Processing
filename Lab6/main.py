import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab6'
tool = SignalToolkit(output_dir='charts')
plt.rcParams["figure.dpi"] = 120
plt.style.use('seaborn-v0_8-whitegrid')


def solve_exercise_1():
    N = 100
    x = np.random.rand(N)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    fig.suptitle("Self-Convolution of a Random Signal", fontsize=16)

    current_signal = x
    axs[0].plot(current_signal)
    axs[0].set_title("Original Signal")
    
    for i in range(1, 4):
        current_signal = np.convolve(current_signal, x)
        axs[i].plot(current_signal)
        axs[i].set_title(f"Iteration {i}")

    for ax in axs: ax.set_ylabel("Amplitude")
    axs[-1].set_xlabel("Samples")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    tool.save_figure(fig, 'exercise_1', lab_name=LAB_NAME)
    plt.close(fig)


def solve_exercise_2():
    def polynomial_product_fft(p_coeffs, q_coeffs):
        result_len = len(p_coeffs) + len(q_coeffs) - 1
        p_fft = np.fft.fft(p_coeffs, result_len)
        q_fft = np.fft.fft(q_coeffs, result_len)
        result_coeffs = np.fft.ifft(p_fft * q_fft).real.round()
        return np.poly1d(result_coeffs)

    p = np.poly1d(np.random.randint(-9, 10, np.random.randint(2, 6)))
    q = np.poly1d(np.random.randint(-9, 10, np.random.randint(2, 6)))
    
    print("p(x):\n", p, "\n\nq(x):\n", q, "\n\n" + "="*40)
    print("\nProduct using direct multiplication:\n", p * q)
    print("\nProduct using FFT convolution:\n", polynomial_product_fft(p.c, q.c))


def solve_exercise_3():
    def rectangular_window(Nw):
        return np.ones(Nw)

    def hanning_window(Nw):
        n = np.arange(Nw)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (Nw - 1)))

    f, A, phi, Nw = 100, 1, 0, 200
    
    time_vector = np.linspace(0, 0.2, Nw, endpoint=False)
    sinusoid = A * np.sin(2 * np.pi * f * time_vector + phi)

    rect_sinusoid = sinusoid * rectangular_window(Nw)
    hann_sinusoid = sinusoid * hanning_window(Nw)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    
    axs[0].plot(time_vector, rect_sinusoid)
    axs[0].set_title("Rectangular Windowed Sinusoid")
    
    axs[1].plot(time_vector, hann_sinusoid, color='C1') 
    axs[1].set_title("Hanning Windowed Sinusoid")

    for ax in axs:
        ax.set_xlabel("Timp [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
    
    plt.tight_layout()
    tool.save_figure(fig, 'exercise_3_windowing', lab_name=LAB_NAME)
    plt.close(fig)


def solve_exercise_4():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "Train.csv")
        x = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=2)[100:172]
    except (FileNotFoundError, OSError):
        return

    fig_b, ax_b = plt.subplots(figsize=(14, 7))
    ax_b.plot(x, label="Raw Signal", color='black', lw=2)
    for w in [5, 9, 13, 17]:
        kernel = np.ones(w) / w
        filtered = np.convolve(x, kernel, "valid")
        ax_b.plot(np.arange((w-1)//2, len(filtered) + (w-1)//2), filtered, label=f"Smoothed (w={w})")
    
    ax_b.set_title("Moving Average Filter")
    ax_b.set_xlabel("Time [hours]")
    ax_b.set_ylabel("Vehicle Count")
    ax_b.grid(True)
    ax_b.legend()
    tool.save_figure(fig_b, 'exercise_4', lab_name=LAB_NAME)
    plt.close(fig_b)
    
    fs = 1/3600 
    Wn = (1/(12*3600)) / (fs/2)
    print(f"Normalized Cutoff Frequency for a 12 hour period: {Wn:.4f}")
    
    order, rp = 5, 5
    b_butter, a_butter = scipy.signal.butter(order, Wn, btype='low')
    b_cheby, a_cheby = scipy.signal.cheby1(order, rp, Wn, btype='low')
    x_butter = scipy.signal.filtfilt(b_butter, a_butter, x)
    x_cheby = scipy.signal.filtfilt(b_cheby, a_cheby, x)
    
    fig_de, ax_de = plt.subplots(figsize=(14, 7))
    ax_de.plot(x, label="Raw Signal", alpha=0.5, color='gray', ls=':')
    ax_de.plot(x_butter, label="Butterworth Filtered", color='green')
    ax_de.plot(x_cheby, label=f"Chebyshev Filtered (rp={rp}dB)", color='red')
    ax_de.set_title("Butterworth vs. Chebyshev Filter (Order 5)")
    ax_de.set_xlabel("Time [hours]")
    ax_de.set_ylabel("Vehicle Count")
    ax_de.grid(True)
    ax_de.legend()
    tool.save_figure(fig_de, 'exercise_4_comparison', lab_name=LAB_NAME)
    plt.close(fig_de)

    fig_f, axes_f = plt.subplots(2, 1, figsize=(12, 10))
    fig_f.suptitle("Effect of Filter Parameters", fontsize=16)
    
    axes_f[0].plot(x, label="Raw Signal", alpha=0.3, color='gray')
    for order_test in [2, 10]:
        b_b, a_b = scipy.signal.butter(order_test, Wn)
        x_b = scipy.signal.filtfilt(b_b, a_b, x)
        axes_f[0].plot(x_b, label=f"Butterworth (Order={order_test})")
    axes_f[0].set_title("Effect of Filter Order")
    axes_f[0].set_xlabel("Time [hours]")
    axes_f[0].set_ylabel("Vehicle Count")
    axes_f[0].legend()
    axes_f[0].grid(True)

    axes_f[1].plot(x, label="Raw Signal", alpha=0.3, color='gray')
    for rp_test in [1, 10]:
        b_c, a_c = scipy.signal.cheby1(5, rp_test, Wn)
        x_c = scipy.signal.filtfilt(b_c, a_c, x)
        axes_f[1].plot(x_c, label=f"Chebyshev (rp={rp_test} dB)")
    axes_f[1].set_title("Effect of Chebyshev Ripple (rp)")
    axes_f[1].set_xlabel("Time [hours]")
    axes_f[1].set_ylabel("Vehicle Count")
    axes_f[1].legend()
    axes_f[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    tool.save_figure(fig_f, 'exercise_4f_parameter_effects', lab_name=LAB_NAME)
    plt.close(fig_f)
    print("Optimal choice: Butterworth filter of order 5-8.")


def run():
    solve_exercise_1()
    solve_exercise_2()
    solve_exercise_3()
    solve_exercise_4()

if __name__ == "__main__":
    run()