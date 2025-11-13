import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab6'
tool = SignalToolkit(output_dir='charts')
plt.rcParams["figure.dpi"] = 120
plt.style.use('seaborn-v0_8-whitegrid')

import numpy as np
import matplotlib.pyplot as plt

def solve_exercise_1():
    bandwitdths = [0.1, 1.0, 4.0]
    for bandwidth in bandwitdths:
        solve_1_different_bandwidth(bandwidth)
        

def solve_1_different_bandwidth(B=0.1):
    t_continuous = np.arange(-3, 3, 0.01)
    x_original = np.sinc(B * t_continuous)**2
    
    sampling_frequencies = [1.0, 1.5, 2.0, 4.0]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i, Fs in enumerate(sampling_frequencies):
        ax = axs[i]
        Ts = 1 / Fs 

        t_sampled = np.arange(-3, 3 + 1e-9, Ts)
        x_sampled = np.sinc(B * t_sampled)**2
        
        ax.stem(t_sampled, x_sampled, linefmt='C1-', markerfmt='C1o', basefmt=' ', label='Eșantioane x[n]')

        sinc_matrix = np.sinc((t_continuous - t_sampled[:, np.newaxis]) / Ts)
        x_reconstructed = np.dot(x_sampled, sinc_matrix)
        
        ax.plot(t_continuous, x_original, 'k', alpha=0.7, label='Original x(t)')
        ax.plot(t_continuous, x_reconstructed, 'g--', label='Reconstruit $\hat{x}(t)$')
        
        ax.set_title(f'$F_s = {Fs:.2f}$ Hz')
        ax.set_xlabel('$t$ [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle=':', which='both')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-0.2, 1.2])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle(f'Funcția $sinc^2({B}t)$, reconstrucția și punctele de eșantionare', fontsize=16, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    tool.save_figure(fig, f'exercise_1b_with_bandwidth_{B}', lab_name=LAB_NAME)
    
    visualize_sinc_components(B)
    
    
def visualize_sinc_components(B=1.0, Fs=2.0):
    Ts = 1 / Fs
    
    t_continuous = np.linspace(-1.5, 1.5, 500)
    
    t_sampled = np.arange(-2 * Ts, 2 * Ts + Ts/2, Ts)
    x_sampled = np.sinc(B * t_sampled)**2

    fig, ax = plt.subplots(figsize=(6, 4))

    for t_n, x_n in zip(t_sampled, x_sampled):
        component_sinc = x_n * np.sinc((t_continuous - t_n) / Ts)
        ax.plot(t_continuous, component_sinc)

    ax.plot(t_sampled, np.zeros_like(t_sampled), 'ko', label='Puncte de eșantionare')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('$t$ [s]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='-', color='gray', alpha=0.15)
    ax.set_xlim(t_continuous.min(), t_continuous.max())
    ax.set_ylim(-0.25, 1.05)

    ax.text(-1.45, 0.9, r'$x[n]\mathrm{sinc}\left(\frac{t-nT_s}{T_s}\right)$', 
            fontsize=16, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    tool.save_figure(fig, f'exercise_1c_with_bandwidth_{B}', lab_name=LAB_NAME)
    


def solve_exercise_2():
    N = 100
    x = np.random.rand(N)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    fig.suptitle("Self-Convolution of a Random Signal", fontsize=16)
    current_signal = x
    current_signal /= np.linalg.norm(current_signal)
    axs[0].plot(current_signal)
    axs[0].set_title("Original Signal")
    
    for i in range(1, 4):
        current_signal = np.convolve(current_signal, current_signal)
        current_signal /= np.linalg.norm(current_signal)
        axs[i].plot(current_signal)
        axs[i].set_title(f"Iteration {i}")

    for ax in axs: ax.set_ylabel("Amplitude")
    axs[-1].set_xlabel("Samples")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    tool.save_figure(fig, 'exercise_2', lab_name=LAB_NAME)
    plt.close(fig)
    

def solve_exercise_2b():
    N = 100
    x = np.random.rand(N)
    start_range = 45
    end_range = 77

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    fig.suptitle("Self-Convolution of a Random Signal", fontsize=16)
    current_signal = np.array([(1 if start_range <= i and i <= end_range else 0) for i in range(N)]).astype("float64")
    current_signal /= np.linalg.norm(current_signal)
    axs[0].plot(current_signal)
    axs[0].set_title("Original Signal")
    
    for i in range(1, 4):
        current_signal = np.convolve(current_signal, current_signal)
        current_signal /= np.linalg.norm(current_signal)
        axs[i].plot(current_signal)
        axs[i].set_title(f"Iteration {i}")

    for ax in axs: ax.set_ylabel("Amplitude")
    axs[-1].set_xlabel("Samples")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    tool.save_figure(fig, 'exercise_2_b', lab_name=LAB_NAME)
    plt.close(fig)


def solve_exercise_3():
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


def compare_signal_types(signal_type='random'):
    N = 20
    d = 5

    if signal_type == 'random':
        x = np.random.rand(N)
        x -= np.mean(x) 
        title_suffix = "Random signal (white noise)"
    else:
        t = np.arange(N)
        x = np.sin(2 * np.pi * t / N) + 0.5 * np.cos(6 * np.pi * t / N)
        title_suffix = "Random sinusoidal signal"

    y = np.roll(x, d)
    
    X_fft = np.fft.fft(x)
    Y_fft = np.fft.fft(y)
    
    convolution_freq = X_fft * Y_fft  
    convolution_time = np.fft.ifft(convolution_freq)
    d_found_conv = np.argmax(np.abs(convolution_time))

    epsilon = 1e-12
    impulse_response_freq = Y_fft / (X_fft + epsilon)
    impulse_response_time = np.fft.ifft(impulse_response_freq)
    d_found_deconv = np.argmax(np.abs(impulse_response_time))
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Shift (d={d}) {title_suffix}", fontsize=16)
    
    axs[0].stem(x, label='x')
    axs[0].stem(y, linefmt='C1--', markerfmt='C1o', label='y')
    axs[0].set_title('Original vs Shifted Signal')
    axs[0].legend()

    axs[1].stem(np.abs(convolution_time))
    axs[1].set_title(f"Convolution -> d = {d_found_conv}. Probably wrong.")

    axs[2].stem(np.abs(impulse_response_time))
    axs[2].set_title(f"Deconvolution -> d = {d_found_deconv}. Probably correct - right method")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    tool.save_figure(fig, f'exercise_4_comparing_methods_for_{signal_type}_signal', lab_name=LAB_NAME)


def solve_exercise_4():
    compare_signal_types(signal_type='random')
    compare_signal_types(signal_type='sinusoidal')    
    

def solve_exercise_5():
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
    tool.save_figure(fig, 'exercise_5_windowing', lab_name=LAB_NAME)
    plt.close(fig)


def solve_exercise_6():
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
    tool.save_figure(fig_b, 'exercise_5', lab_name=LAB_NAME)
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
    tool.save_figure(fig_de, 'exercise_6_comparison', lab_name=LAB_NAME)
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
    tool.save_figure(fig_f, 'exercise_6f_parameter_effects', lab_name=LAB_NAME)
    plt.close(fig_f)
    print("Optimal choice: Butterworth filter of order 5-8.")


def run():
    solve_exercise_1()
    solve_exercise_2()
    solve_exercise_2b()
    solve_exercise_3()
    solve_exercise_4()
    solve_exercise_5()
    solve_exercise_6()

if __name__ == "__main__":
    run()