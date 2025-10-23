import numpy as np
import matplotlib.pyplot as plt
import os
from signal_processing_core import SignalToolkit


class TransformataFourier:
    def __init__(self):
        self.tool = SignalToolkit()

    def rezolva_exercitiul_1(self, N=20):
        F = self.tool.create_fourier_matrix(N)
        # use toolkit to plot and save the Fourier matrix rows
        self.tool.plot_fourier_matrix(F, lab_name='Lab3')

        # verify unitary property
        produs = F.conj().T.dot(F)
        matrice_identitate_N = N * np.identity(N)
        este_unitara = np.allclose(produs, matrice_identitate_N)
        print(f"Verificare unitaritate (F^H * F == N * I): {este_unitara}")


    def rezolva_exercitiul_2(self):
        fs = 500  # Frecvența de eșantionare (Hz)
        T = 1.0  # Durata semnalului (secunde)
        N_esantioane = int(fs * T)
        t = np.linspace(0, T, N_esantioane, endpoint=False)
        f_semnal = 7
        # use toolkit to generate and plot
        t, x = self.tool.generate_signal(self.tool.SignalType.SINE, freq=f_semnal, duration=T, sampling_freq=fs)
        fig1, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(t, x)
        ax.set_title('Semnal Sinusoidal în Domeniul Timp')
        ax.set_xlabel('Timp (s)')
        ax.set_ylabel('Amplitudine')
        ax.grid(True)

        # complex plane scatter
        y = x * np.exp(-2j * np.pi * f_semnal * t)
        distanta = np.abs(x)
        fig_complex, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        scatter = ax2.scatter(y.real, y.imag, c=distanta, cmap='viridis')
        ax2.set_title('Planul Complex')
        ax2.set_xlabel('Partea Reala')
        ax2.set_ylabel('Partea Imaginara')
        ax2.axis('equal')
        ax2.grid(True)
        fig_complex.colorbar(scatter, ax=ax2, label='Distanta de la origine (|x[n]|)')
        self.tool.save_figure(fig1, 'exercitiul_2_figura_1_time', lab_name='Lab3')
        self.tool.save_figure(fig_complex, 'exercitiul_2_figura_1_complex', lab_name='Lab3')

        frecvente_infasurare = [2, 5, f_semnal, 15]

        fig2, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig2.suptitle(f'Frecventa de infasurare (ω) pentru un semnal de {f_semnal}Hz', fontsize=16)

        for i, omega in enumerate(frecvente_infasurare):
            ax = axs.flatten()[i]
            z = x * np.exp(-2j * np.pi * omega * t)
            distanta_z = np.abs(x)
            scatter = ax.scatter(z.real, z.imag, c=distanta_z, cmap='plasma', s=10)
            ax.set_title(f'ω = {omega} Hz')
            ax.set_xlabel('Partea Reală')
            ax.set_ylabel('Partea Imaginară')
            ax.axis('equal')
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.tool.save_figure(fig2, 'exercitiul_2_figura_2', lab_name='Lab3')
        plt.show()

    def _calculeaza_dft_manual(self, x, fs, N):
        X = []
        frecvente = np.linspace(0, fs / 2, N // 2)
        n_vector = np.arange(N)

        for omega in frecvente:
            exp_term = np.exp(-2j * np.pi * omega * n_vector / fs)
            X_omega = np.sum(x * exp_term)
            X.append(X_omega)

        return frecvente, np.abs(X)

    def rezolva_exercitiul_3(self):
        fs = 150  # Frecvența de eșantionare (trebuie să fie > 2 * f_max)
        T = 2.0
        N = int(fs * T)
        t = np.linspace(0, T, N, endpoint=False)

        f1, amp1 = 12, 1.0
        f2, amp2 = 30, 1.5
        f3, amp3 = 55, 0.8

        x = (amp1 * np.cos(2 * np.pi * f1 * t) +
             amp2 * np.cos(2 * np.pi * f2 * t) +
             amp3 * np.cos(2 * np.pi * f3 * t))

        frecvente, magnitudine_dft = self._calculeaza_dft_manual(x, fs, N)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Figura 3: Transformata Fourier a unui Semnal Compus', fontsize=16)

        ax1.plot(t, x)
        ax1.set_title('Semnal Compus în Domeniul Timp')
        ax1.set_xlabel('Timp (s)')
        ax1.set_ylabel('x(t)')
        ax1.grid(True)
        ax1.set_xlim(0, T)

        ax2.plot(frecvente, magnitudine_dft)
        ax2.set_title('Modulul Transformatei Fourier')
        ax2.set_xlabel('Frecvența (Hz)')
        ax2.set_ylabel('|X(ω)|')
        ax2.grid(True)
        ax2.set_xticks([f1, f2, f3])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.tool.save_figure(fig, 'exercitiul_3_figura_3', lab_name='Lab3')
        plt.show()

    def ruleaza_toate_exercitiile(self):
        self.rezolva_exercitiul_1()
        self.rezolva_exercitiul_2()
        self.rezolva_exercitiul_3()


if __name__ == '__main__':
    laborator = TransformataFourier()
    laborator.ruleaza_toate_exercitiile()

