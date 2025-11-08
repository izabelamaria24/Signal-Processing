import numpy as np
import matplotlib.pyplot as plt
from signal_processing_core import SignalToolkit

tool = SignalToolkit()

def plot_1d_signal(name, signal_type, freq, sampling_freq=8000, duration=1.0, lab_name='Lab1', filename=None):
    t, x = tool.generate_signal(getattr(tool.SignalType, signal_type.name), freq=freq, sampling_freq=sampling_freq, duration=duration)
    tool.plot_signal(t, x, title=name, lab_name=lab_name, filename=filename)


def plot_and_save_2d(data, title, filename, colormap):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap=colormap)
    plt.title(title)
    plt.colorbar()
    tool.save_figure(fig, filename, lab_name='Lab1')
    plt.show()


def run():
    signals_to_plot = [
        {"name": "a) Sin 400 Hz with 1600 samples", "type": tool.SignalType.SINE, "freq": 400, "duration": 1600/8000, "filename": 'a_sinus_400Hz'},
        {"name": "b) Sin 800 Hz, duration 3s", "type": tool.SignalType.SINE, "freq": 800, "duration": 3, "filename": 'b_sinus_800Hz'},
        {"name": "c) Sawtooth 240 Hz", "type": tool.SignalType.SAWTOOTH, "freq": 240, "duration": 0.02, "filename": 'c_sawtooth_240Hz'},
        {"name": "d) Square 300 Hz", "type": tool.SignalType.SQUARE, "freq": 300, "duration": 0.01, "filename": 'd_square_300Hz'}
    ]

    for item in signals_to_plot:
        plot_1d_signal(item['name'], item['type'], item['freq'], sampling_freq=8000, duration=item['duration'], lab_name='Lab1', filename=item['filename'])

    random_signal_2d = np.random.rand(128, 128)
    plot_and_save_2d(random_signal_2d, 'e) Random 2D Signal (128x128)', 'e_random_2D', 'gray')

    custom_signal_2d = np.zeros((128, 128))
    gradient = np.linspace(0, 1, 128)
    custom_signal_2d[:, :] = gradient
    plot_and_save_2d(custom_signal_2d, 'f) Custom 2D Signal (Horizontal Gradient)', 'f_custom_2D', 'plasma')


if __name__ == '__main__':
    run()