import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab8'
tool = SignalToolkit(output_dir='charts')

def generate_time_series():
    N = 1000

    t = np.linspace(0, 1, N)
    trend = t ** 2 + 1
    season = np.sin(2 * 20 * np.pi * t) + np.sin(3 * 15 * np.pi * t)
    noise = np.random.normal(0, 0.2, N)
    
    return trend, season, noise


def solve1():
    trend, season, noise = generate_time_series()    

    fig, axs = plt.subplots(4)
    axs[0].plot(trend)
    axs[1].plot(season)
    axs[2].plot(noise)
    axs[3].plot(trend + season + noise)
    plt.tight_layout()
    plt.show()
    tool.save_figure(fig, f'ex1', lab_name=LAB_NAME)


def solve2():
    trend, season, noise = generate_time_series()
    time_series = trend + season + noise
    
    numpy_autocorrelation = np.correlate(time_series, time_series, mode='full')
    numpy_autocorrelation /= np.max(numpy_autocorrelation)
    
    fig, axs = plt.subplots(3)
    axs[0].plot(numpy_autocorrelation)
    
    numpy_autocorrelation = numpy_autocorrelation[len(numpy_autocorrelation)//2:]

    axs[1].plot(numpy_autocorrelation)
    
    
    def own_autocorrelate(time_series, i):
        N = len(time_series)
        mean = np.mean(time_series)
        
        return np.sum((time_series[i:] - mean) * (time_series[:N-i] - mean)) / (N - i)
    
    autocor = []
    for i in range(0, 1000):
        autocor.append(own_autocorrelate(time_series, i))
        
    axs[2].plot(autocor)
    autocor = autocor[len(autocor) //2:]
    
        
    plt.tight_layout()
    plt.show()
    tool.save_figure(fig, f'ex2-np_autocorrelation', lab_name=LAB_NAME)    
    
    
def solve3():
    trend, season, noise = generate_time_series()
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
    
    tool.save_figure(fig, f'ex3-AR', lab_name=LAB_NAME)


def ar_predict_one_step(train_data, p):
    y = train_data[p:]
    m = len(y)
    Y = np.zeros((m, p))
    for i in range(p):
        Y[:, i] = train_data[i:i + m]
    
    try:
        big_gamma = Y.T @ Y
        small_gamma = Y.T @ y
        x_star = np.linalg.inv(big_gamma) @ small_gamma
    except np.linalg.LinAlgError:
        return None

    last_p = train_data[-p:]
    prediction = x_star @ last_p
    return prediction


def solve4():
    trend, season, noise = generate_time_series()
    time_series = trend + season + noise
    
    train_size = int(len(time_series) * 0.8)
    train_data, validation_data = time_series[:train_size], time_series[train_size:]
    
    p_values = range(1, 151)  
    errors = []

    for p in p_values:
        predictions = []
        history = list(train_data)
        
        for t in range(len(validation_data)):
            pred = ar_predict_one_step(np.array(history), p)
            if pred is not None:
                predictions.append(pred)
                history.append(validation_data[t])
            else:
                predictions = []
                break
        
        if len(predictions) == len(validation_data):
            error = mean_squared_error(validation_data, predictions)
            errors.append(error)
        else:
            errors.append(np.inf)

    best_p_index = np.argmin(errors)
    best_p = p_values[best_p_index]
    min_error = errors[best_p_index]
    
    print(f"Best p = {best_p} cu MSE: {min_error}")
    
    fig, ax = plt.subplots()
    ax.plot(p_values, errors)
    ax.set_xlabel("p ")
    ax.set_ylabel("MSE")
    ax.set_title("Hyperparameter tunning")
    ax.axvline(x=best_p, color='r', linestyle='--', label=f'Cel mai bun p = {best_p}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    tool.save_figure(fig, 'ex4-hyperparameter-tuning', lab_name=LAB_NAME)


def run():
    solve1()
    solve2()
    solve3()
    solve4()

if __name__ == "__main__":
    run()