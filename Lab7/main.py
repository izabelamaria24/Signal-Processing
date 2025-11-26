from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt


def solve_ex_1():
    n1, n2 = 64, 64
    
    def plot_img_and_spectrum(X=None, Y=None, name="Title"):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        if Y is None:
            Y = 20 * np.log10(abs(np.fft.fft2(X)) + 1e-69)
        elif X is None:
            X = np.fft.ifft2(Y)

        axs[1].imshow(Y), axs[1].set_title("Spectrul Functiei")
        fig.colorbar(axs[1].imshow(Y), ax=axs[1])

        axs[0].imshow(np.real(X)), axs[0].set_title("Functia")
        fig.colorbar(axs[0].imshow(np.real(X)), ax=axs[0])

        plt.show()


    def create_and_plot_from_signal(func):
        i, j = np.indices((n1, n2))
        x = func(i, j)
        plot_img_and_spectrum(X=x)


    create_and_plot_from_signal(lambda i, j: np.sin(2 * np.pi * i + 3 * np.pi * j))
    create_and_plot_from_signal(lambda i, j: np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j))    


def run():
    solve_ex_1()


if __name__ == "__main__":
    run()

