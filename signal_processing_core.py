import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from enum import Enum


class SignalToolkit:
    def __init__(self, output_dir='charts'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def lab_asset_path(self, lab_name):
        path = os.path.join(self.output_dir, lab_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _sanitize_filename(self, name):
        # Replace Windows-illegal characters: <>:"/\|?*
        illegal = '<>:"/\\|?*'
        for ch in illegal:
            name = name.replace(ch, '_')
        return name.strip().rstrip('.')

    def save_figure(self, fig, name, lab_name=None):
        safe_name = self._sanitize_filename(name)
        if lab_name:
            folder = self.lab_asset_path(lab_name)
        else:
            folder = self.output_dir
        # Absolute paths to avoid any cwd ambiguity on some platforms
        png_path = os.path.abspath(os.path.join(folder, f"{safe_name}.png"))
        pdf_path = os.path.abspath(os.path.join(folder, f"{safe_name}.pdf"))
        try:
            fig.savefig(png_path)
            fig.savefig(pdf_path)
            print(f"Saved figure: {png_path} and {pdf_path}")
        except OSError as e:
            # Fallback to a generic safe filename
            fallback = self._sanitize_filename(f"{safe_name}_figure")
            png_fallback = os.path.abspath(os.path.join(folder, f"{fallback}.png"))
            pdf_fallback = os.path.abspath(os.path.join(folder, f"{fallback}.pdf"))
            fig.savefig(png_fallback)
            fig.savefig(pdf_fallback)
            print(f"Saved figure with fallback names due to error ({e}): {png_fallback} and {pdf_fallback}")
            return png_fallback
        return png_path

    def create_fourier_matrix(self, N=8):
        n = np.arange(N)
        k = n.reshape((N, 1))
        F = np.exp(-2j * np.pi * k * n / N)
        return F


    def generate_signal(self, signal_type, freq, amplitude=1.0, phase=0.0, sampling_freq=8000, duration=None, num_samples=None):
        if duration is not None:
            t = np.arange(0, duration, 1.0 / sampling_freq)
        elif num_samples is not None:
            n = np.arange(num_samples)
            t = n / sampling_freq
        else:
            raise ValueError('Either duration or num_samples must be provided')

        omega = 2 * np.pi * freq
        if signal_type == self.SignalType.SINE:
            x = amplitude * np.sin(omega * t + phase)
        elif signal_type == self.SignalType.COSINE:
            x = amplitude * np.cos(omega * t + phase)
        elif signal_type == self.SignalType.SAWTOOTH:
            x = amplitude * 2 * (t * freq - np.floor(0.5 + t * freq))
        elif signal_type == self.SignalType.SQUARE:
            x = amplitude * np.sign(np.sin(omega * t + phase))
        else:
            raise ValueError('Unsupported signal type')

        return t, x

    def apply_fourier_matrix(self, F, x):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('x must be a 1-D array')
        return F.dot(x)

    def _is_power_of_two(self, n):
        return n > 0 and (n & (n - 1)) == 0

    def _fft_ct(self, x):
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]
        if N <= 1:
            return x
        if N % 2 != 0:
            # Fallback to direct DFT if N is not even (should not happen for power-of-two sizes)
            n = np.arange(N)
            k = n.reshape((N, 1))
            F = np.exp(-2j * np.pi * k * n / N)
            return F.dot(x)
        X_even = self._fft_ct(x[::2])
        X_odd = self._fft_ct(x[1::2])
        twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N) * X_odd
        return np.concatenate([X_even + twiddle[: N // 2], X_even - twiddle[: N // 2]])

    def fft(self, x):
        """
        Compute the FFT of a 1-D array using a simple Cooley–Tukey radix-2 algorithm.
        Requires len(x) to be a power of two.
        """
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]
        if not self._is_power_of_two(N):
            raise ValueError('fft requires input length to be a power of two')
        return self._fft_ct(x)

    def plot_fourier_matrix(self, F, lab_name=None):
        N = F.shape[0]
        fig, axes = plt.subplots(N, 1, figsize=(6, 2 * N))
        if N == 1:
            axes = [axes]
        n = np.arange(F.shape[1])
        for i, ax in enumerate(axes):
            ax.plot(n, F[i].real, label='real')
            ax.plot(n, F[i].imag, label='imag', linestyle='--')
            ax.set_title(f'Row {i}')
            ax.set_xlabel('n')
            ax.set_ylabel('value')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        if lab_name:
            self.save_figure(fig, 'fourier_matrix', lab_name=lab_name)
        else:
            plt.show()

    def plot_signal(self, t, x, title=None, lab_name=None, filename=None):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, x)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        if title:
            ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        if lab_name and filename:
            self.save_figure(fig, filename, lab_name=lab_name)
        else:
            plt.show()

    def create_FH(self, F):
        FH = F.conj().T
        N = F.shape[0]
        invF = FH / N
        return invF

    def generate_sinusoidal_signal(self, frequency, amplitude, phase, sampling_rate, duration):
        t = np.arange(0, duration, 1.0 / sampling_rate)
        x = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return t, x

    def plot_phasor(self, t, x, omega=1, amplitude=None, save_gif=None, downsample=5, fps=25):
        if amplitude is None:
            amplitude = np.max(np.abs(x)) if len(x) else 1.0

        z = x * np.exp(1j * 2 * np.pi * omega * t)
        N = len(t)

        if not save_gif:
            fig, ax = plt.subplots(figsize=(7, 7))
            theta = np.linspace(0, 2 * np.pi, 400)
            ax.plot(amplitude * np.cos(theta), amplitude * np.sin(theta), 'k--', alpha=0.4)
            ax.plot(z.real, z.imag, color='cornflowerblue', linewidth=2)
            ax.scatter(z.real, z.imag, s=18, c=np.linspace(0, 1, N), cmap='viridis')
            if N:
                ax.plot([0, z.real[-1]], [0, z.imag[-1]], 'r-')
                ax.plot(z.real[-1], z.imag[-1], 'ro')
            ax.set_aspect('equal')
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
            ax.set_title(f'omega = {omega}')
            ax.grid(True)
            lim = amplitude * 1.1
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            plt.show()
            return None

        indices = np.arange(0, N, downsample)
        if indices.size == 0 and N > 0:
            indices = np.array([N - 1])
        if indices.size and indices[-1] != N - 1:
            indices = np.append(indices, N - 1)

        fig, ax = plt.subplots(figsize=(7, 7))
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(amplitude * np.cos(theta), amplitude * np.sin(theta), 'k--', alpha=0.4)
        ax.axhline(0, color='black', lw=0.75)
        ax.axvline(0, color='black', lw=0.75)
        ax.set_aspect('equal', adjustable='box')
        lim = amplitude * 1.1
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.grid(True)

        traj_line, = ax.plot([], [], color='cornflowerblue', linewidth=2)
        current_dot, = ax.plot([], [], 'ro')
        radial_line, = ax.plot([], [], 'r-', linewidth=2)

        def init():
            traj_line.set_data([], [])
            current_dot.set_data([], [])
            radial_line.set_data([], [])
            return traj_line, current_dot, radial_line

        def update(frame_idx):
            i = indices[frame_idx]
            traj_line.set_data(z.real[:i+1], z.imag[:i+1])
            current_dot.set_data(z.real[i], z.imag[i])
            radial_line.set_data([0, z.real[i]], [0, z.imag[i]])
            return traj_line, current_dot, radial_line

        ani = FuncAnimation(fig, update, frames=len(indices), init_func=init, blit=True, repeat=False)

        if save_gif:
            out_name = save_gif if isinstance(save_gif, str) else f'phasor_omega_{omega}.gif'
            writer = PillowWriter(fps=fps)
            ani.save(out_name, writer=writer)
            print(f"Saved GIF: {out_name}")

        plt.show()


class SignalType(Enum):
    SINE = 0
    SAWTOOTH = 1
    SQUARE = 2
    COSINE = 3

SignalToolkit.SignalType = SignalType

