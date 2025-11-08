import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
try:
    import librosa 
except Exception:
    librosa = None

from signal_processing_core import SignalToolkit

LAB_NAME = "Lab4"

def solve_ex1(toolkit):
    vector_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    my_dft_times = []
    my_fft_times = []
    numpy_fft_times = []

    def time_call(callable_fn, repeats=3):
        import time as _t
        best = float('inf')
        for _ in range(repeats):
            start = _t.perf_counter()
            callable_fn()
            elapsed = _t.perf_counter() - start
            if elapsed < best:
                best = elapsed
        return max(best, 1e-9)

    for N in vector_sizes:
        print(f"Testing for N = {N}...")
        x = np.random.rand(N) + 1j * np.random.rand(N)

        F = toolkit.create_fourier_matrix(N)
        my_dft_times.append(time_call(lambda: toolkit.apply_fourier_matrix(F, x), repeats=1))

        my_fft_times.append(time_call(lambda: toolkit.fft(x), repeats=3))

        numpy_fft_times.append(time_call(lambda: np.fft.fft(x), repeats=3))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(vector_sizes, my_dft_times, 'o-', label='Custom Implementation (DFT Matrix O(N^2))')
    ax.plot(vector_sizes, my_fft_times, '^-', label='Custom Implementation (FFT O(N log N))')
    ax.plot(vector_sizes, numpy_fft_times, 's-', label='numpy.fft.fft (FFT O(N log N))')
    ax.set_xlabel('Vector Size (N)')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time Comparison: DFT vs FFT')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    toolkit.save_figure(fig, "ex1_time_comparison", lab_name=LAB_NAME)
    plt.show()


def solve_ex2(toolkit):
    f0 = 100
    fs_cont = 2000 
    duration = 0.03
    fs_sample = 120 
    
    f_alias1 = fs_sample - f0 
    f_alias2 = fs_sample + f0 
    
    t_cont, x_cont = toolkit.generate_sinusoidal_signal(f0, 1, 0, fs_cont, duration)
    _, x_alias1 = toolkit.generate_sinusoidal_signal(f_alias1, 1, 0, fs_cont, duration)
    _, x_alias2 = toolkit.generate_sinusoidal_signal(f_alias2, 1, 0, fs_cont, duration)
    
    t_sample, x_sample = toolkit.generate_sinusoidal_signal(f0, 1, 0, fs_sample, duration)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Aliasing (f0={f0} Hz, fs={fs_sample} Hz)', fontsize=16)

    axes[0].plot(t_cont, x_cont, 'b-', label=f'Original ({f0} Hz)')
    axes[0].plot(t_sample, x_sample, 'yo', markersize=8, label='Samples')
    axes[1].plot(t_cont, x_alias1, 'r-', label=f'Alias ({abs(f_alias1)} Hz)')
    axes[1].plot(t_sample, x_sample, 'yo', markersize=8)
    axes[2].plot(t_cont, x_alias2, 'g-', label=f'Alias ({f_alias2} Hz)')
    axes[2].plot(t_sample, x_sample, 'yo', markersize=8)
    
    for ax in axes:
        ax.legend()
        ax.grid(True)
    axes[2].set_xlabel('Time (s)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    toolkit.save_figure(fig, "ex2_aliasing", lab_name=LAB_NAME)
    print(f"Three sinusoids ({f0}Hz, {abs(f_alias1)}Hz, {f_alias2}Hz) produce the same samples at fs={fs_sample}Hz.\n")
    plt.show()


def solve_ex3(toolkit):
    f0 = 100
    fs_cont = 2000
    duration = 0.03
    fs_sample = 250 
    
    f_alias1, f_alias2 = 20, 220

    t_cont, x_cont = toolkit.generate_sinusoidal_signal(f0, 1, 0, fs_cont, duration)
    _, x_alias1 = toolkit.generate_sinusoidal_signal(f_alias1, 1, 0, fs_cont, duration)
    _, x_alias2 = toolkit.generate_sinusoidal_signal(f_alias2, 1, 0, fs_cont, duration)
    
    t_sample, x_sample = toolkit.generate_sinusoidal_signal(f0, 1, 0, fs_sample, duration)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Correct Sampling (f0={f0} Hz, fs={fs_sample} Hz)', fontsize=16)

    axes[0].plot(t_cont, x_cont, 'b-', label=f'Original ({f0} Hz)')
    axes[0].plot(t_sample, x_sample, 'yo', markersize=8, label='Samples')
    axes[1].plot(t_cont, x_alias1, 'r-', label=f'Signal {f_alias1} Hz')
    axes[1].plot(t_sample, x_sample, 'yo', markersize=8, label='Samples do NOT match')
    axes[2].plot(t_cont, x_alias2, 'g-', label=f'Signal {f_alias2} Hz')
    axes[2].plot(t_sample, x_sample, 'yo', markersize=8, label='Samples do NOT match')

    for ax in axes:
        ax.legend()
        ax.grid(True)
    axes[2].set_xlabel('Time (s)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    toolkit.save_figure(fig, "ex3_correct_sampling", lab_name=LAB_NAME)
    print(f"With fs={fs_sample}Hz (> 2*f0), the {f0}Hz signal is uniquely represented by the samples; {f_alias1}Hz and {f_alias2}Hz do not match the samples (no alias).")
    plt.show()


def solve_ex6(toolkit):
    if librosa is None:
        print("librosa not installed or unavailable — skipping spectrograms in ex6.")
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))

    m4a_pattern = os.path.join(script_dir, '*.m4a')
    wav_pattern = os.path.join(script_dir, '*.wav')

    audio_files = glob.glob(m4a_pattern) + glob.glob(wav_pattern)

    for audio_filepath in audio_files:
        try:
            signal, fs = librosa.load(audio_filepath, sr=None, mono=True)
        except Exception as e:
            print(f"Could not load file {audio_filepath}. Error: {e}")
            continue
            
        N = len(signal)
        window_size = 1024
        overlap = window_size // 2
        step = window_size - overlap
        
        num_frames = 1 + (N - window_size) // step
        
        spectrogram_matrix = []
        for i in range(num_frames):
            frame = signal[i*step : i*step + window_size]
            windowed_frame = frame * np.hanning(window_size)
            fft_result = np.fft.fft(windowed_frame)
            fft_magnitude = np.abs(fft_result[:window_size // 2])
            spectrogram_matrix.append(fft_magnitude)

        spectrogram_matrix = np.array(spectrogram_matrix).T
        spectrogram_matrix[spectrogram_matrix == 0] = 1e-10
        db_spectrogram = 20 * np.log10(spectrogram_matrix)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        time_axis = np.arange(num_frames) * step / fs
        freq_axis = np.arange(window_size // 2) * fs / window_size
        img = ax.pcolormesh(time_axis, freq_axis, db_spectrogram, shading='gouraud', cmap='viridis')
        
        vowel_name = os.path.splitext(os.path.basename(audio_filepath))[0]
        ax.set_title(f"Spectrogram for Vowel '{vowel_name}'")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Magnitude (dB)')
        
        toolkit.save_figure(fig, f"ex6_spectrogram_{vowel_name}", lab_name=LAB_NAME)
        plt.show()


def solve_ex7():
    P_signal_dB = 90
    SNR_dB = 80
    
    P_noise_dB = P_signal_dB - SNR_dB
    
    print(f"P_noise_dB = {P_signal_dB} dB - {SNR_dB} dB = {P_noise_dB} dB.\n")


def run():
    toolkit = SignalToolkit()
    solve_ex1(toolkit)
    solve_ex2(toolkit)
    solve_ex3(toolkit)
    solve_ex6(toolkit)
    solve_ex7()


if __name__ == "__main__":
    run()
