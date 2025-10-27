import numpy as np
import matplotlib.pyplot as plt
import os
from signal_processing_core import SignalToolkit


class FourierTransform:
    def __init__(self):
        self.tool = SignalToolkit()

    def solve_exercise_1(self, N=20):
        F = self.tool.create_fourier_matrix(N)
        self.tool.plot_fourier_matrix(F, lab_name='Lab3')

        product = F.conj().T.dot(F)
        identity_matrix_N = N * np.identity(N)
        is_unitary = np.allclose(product, identity_matrix_N)
        print(f"Unitary check (F^H * F == N * I): {is_unitary}")


    def solve_exercise_2(self):
        fs = 500
        T = 1.0
        num_samples = int(fs * T)
        t = np.linspace(0, T, num_samples, endpoint=False)
        signal_freq = 7

        t, x = self.tool.generate_signal(self.tool.SignalType.SINE, freq=signal_freq, duration=T, sampling_freq=fs)
        fig1, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(t, x)
        ax.set_title('Sinusoidal Signal in Time Domain')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        y = x * np.exp(-2j * np.pi * signal_freq * t)
        distance = np.abs(x)
        fig_complex, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        scatter = ax2.scatter(y.real, y.imag, c=distance, cmap='viridis')
        ax2.set_title('Complex Plane')
        ax2.set_xlabel('Real Part')
        ax2.set_ylabel('Imaginary Part')
        ax2.axis('equal')
        ax2.grid(True)
        fig_complex.colorbar(scatter, ax=ax2, label='Distance from origin (|x[n]|)')
        self.tool.save_figure(fig1, 'exercise_2_figure_1_time', lab_name='Lab3')
        self.tool.save_figure(fig_complex, 'exercise_2_figure_1_complex', lab_name='Lab3')

        wrapping_frequencies = [2, 5, signal_freq, 15]

        fig2, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig2.suptitle(f'Wrapping Frequency (ω) for a {signal_freq}Hz signal', fontsize=16)

        for i, omega in enumerate(wrapping_frequencies):
            ax = axs.flatten()[i]
            z = x * np.exp(-2j * np.pi * omega * t)
            distance_z = np.abs(x)
            scatter = ax.scatter(z.real, z.imag, c=distance_z, cmap='plasma', s=10)
            ax.set_title(f'ω = {omega} Hz')
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.axis('equal')
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.tool.save_figure(fig2, 'exercise_2_figure_2', lab_name='Lab3')
        plt.show()

    def _calculate_dft_manual(self, x, fs, N):
        X = []
        frequencies = np.linspace(0, fs / 2, N // 2)
        n_vector = np.arange(N)

        for omega in frequencies:
            exp_term = np.exp(-2j * np.pi * omega * n_vector / fs)
            X_omega = np.sum(x * exp_term)
            X.append(X_omega)

        return frequencies, np.abs(X)

    def solve_exercise_3(self):
        fs = 150
        T = 2.0
        N = int(fs * T)
        t = np.linspace(0, T, N, endpoint=False)

        f1, amp1 = 12, 1.0
        f2, amp2 = 30, 1.5
        f3, amp3 = 55, 0.8

        x = (amp1 * np.cos(2 * np.pi * f1 * t) +
             amp2 * np.cos(2 * np.pi * f2 * t) +
             amp3 * np.cos(2 * np.pi * f3 * t))

        frequencies, dft_magnitude = self._calculate_dft_manual(x, fs, N)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Figure 3: Fourier Transform of a Composite Signal', fontsize=16)

        ax1.plot(t, x)
        ax1.set_title('Composite Signal in Time Domain')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('x(t)')
        ax1.grid(True)
        ax1.set_xlim(0, T)

        ax2.plot(frequencies, dft_magnitude)
        ax2.set_title('Fourier Transform Magnitude')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('|X(ω)|')
        ax2.grid(True)
        ax2.set_xticks([f1, f2, f3])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.tool.save_figure(fig, 'exercise_3_figure_3', lab_name='Lab3')
        plt.show()

    def run_all_exercises(self):
        self.solve_exercise_1()
        self.solve_exercise_2()
        self.solve_exercise_3()


if __name__ == '__main__':
    lab = FourierTransform()
    lab.run_all_exercises()