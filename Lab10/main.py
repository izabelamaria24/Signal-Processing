import numpy as np
from matplotlib import pyplot as plt

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab10'
tool = SignalToolkit(output_dir='charts')

def solve2():
    trend, season, noise = tool.generate_time_series()
    time_series = trend + season + noise
    p = 75
    
    def ar(ts, p):
        train_data = ts[:-p]
        test_data = ts[-p:]
        y = train_data[p:]
        m = len(y)

        Y = np.zeros((m, p))
        for i in range(p):
            Y[:, i] = train_data[i:i-p]

        big_gamma = Y.T @ Y 
        small_gamma = Y.T @ y
        x_star = np.linalg.inv(big_gamma) @ small_gamma

        pred = []
        for i in range(p):
            last_p = np.append(train_data, pred)[-p:]
            pred = np.append(pred, x_star @ last_p)

        return train_data, test_data, pred
    
    
    fig, axs = plt.subplots(4)
    
    p_values = [(2, 0), (50, 1), (75, 2), (125, 3)]
    for (p, index) in p_values:
        N = len(time_series)
        train_data, test_data, pred  = ar(time_series, p)
        x_pred = np.arange(len(train_data), len(train_data) + len(pred))
        
        axs[index].plot(train_data, label=f"Train data for p = {p}")
        axs[index].plot(x_pred, pred, "r-", label=f"Prediction for p = {p}", alpha=0.8)
        axs[index].plot(x_pred, test_data, "g-", label=f"Real for p = {p}", alpha=0.8)
        axs[index].legend()
    
    
    plt.tight_layout()
    plt.show()
    
    tool.save_figure(fig, f'ex2-AR', lab_name=LAB_NAME)


def run():
    solve2()

if __name__ == "__main__":
    run()