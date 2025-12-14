import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm
import itertools
from concurrent.futures import ThreadPoolExecutor

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab9'
tool = SignalToolkit(output_dir='charts')

def solve2():
    trend, season, noise = tool.generate_time_series()
    time_series = trend + season + noise
    
    seasonal_period = 50 

    ses_fixed = SimpleExpSmoothing(time_series).fit(smoothing_level=0.2, optimized=False)
    pred_ses_fixed = ses_fixed.fittedvalues
    
    ses_opt = SimpleExpSmoothing(time_series).fit(optimized=True)
    pred_ses_opt = ses_opt.fittedvalues

    holt_fixed = Holt(time_series).fit(smoothing_level=0.2, smoothing_trend=0.1, optimized=False)
    pred_holt_fixed = holt_fixed.fittedvalues
    
    holt_opt = Holt(time_series).fit(optimized=True)
    pred_holt_opt = holt_opt.fittedvalues

    hw_fixed = ExponentialSmoothing(time_series, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit(
        smoothing_level=0.2, smoothing_trend=0.1, smoothing_seasonal=0.1, optimized=False
    )
    pred_hw_fixed = hw_fixed.fittedvalues
    
    hw_opt = ExponentialSmoothing(time_series, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit(optimized=True)
    pred_hw_opt = hw_opt.fittedvalues
    
    fig = plt.figure(figsize=(14, 8))
    
    plt.plot(time_series, label='Original data', color='black', alpha=0.5)
    
    plt.plot(pred_ses_opt, label='SES (Simple) Optim', linestyle='--')
    plt.plot(pred_holt_opt, label='Holt (Double) Optim', linestyle='--')
    plt.plot(pred_hw_opt, label='Holt-Winters (Triple) Optim', color='red', linewidth=2)
    
    plt.title('Comparison of Exponential Smoothing Methods (Optimized)')
    plt.legend()
    plt.grid(True)
    
    tool.save_figure(fig, 'ex2-exponential-smoothing', lab_name=LAB_NAME)
    
    
def solve3():
    trend, season, noise = tool.generate_time_series()
    time_series = trend + season + noise
    
    q = 50  
    
    ma_series_manual = np.zeros_like(time_series)
    errors = np.zeros_like(time_series)
    
    for i in range(q, len(time_series)):
        window = time_series[i-q : i] 
        mean_val = np.mean(window)
        
        ma_series_manual[i] = mean_val
        
        errors[i] = time_series[i] - mean_val

    ts_pandas = pd.Series(time_series)
    ma_series_pandas = ts_pandas.rolling(window=q).mean()
    ma_series_pandas = ma_series_pandas.fillna(0).to_numpy()

    fig = plt.figure(figsize=(14, 8))
    
    plt.plot(time_series, label='Original series', color='lightgray')
    plt.plot(ma_series_pandas, label=f'Moving Average (MA) q={q}', color='blue', linewidth=2)
    
    plt.plot(errors, label='Errors (Deviation)', color='red', alpha=0.3)
    
    plt.title(f'MA Model (Moving Average) with horizon q={q}')
    plt.legend()
    plt.grid(True)
    
    tool.save_figure(fig, 'ex3-moving-average', lab_name=LAB_NAME)
    plt.show()


def solve4():    
    trend, season, noise = tool.generate_time_series()
    time_series = trend + season + noise
    
    p_values = range(0, 10) 
    q_values = range(0, 10) 
    pq_combinations = list(itertools.product(p_values, q_values))

    print(f"Evaluating {len(pq_combinations)} ARMA models with (ARIMA with d=0)...")

    def evaluate_arima(pq):
        try:
            model = sm.tsa.ARIMA(time_series, order=(pq[0], 0, pq[1])).fit()
            return (model.aic, pq)
        except:
            return (float("inf"), pq)

    results = []
    with ThreadPoolExecutor() as executor:
        for result in executor.map(evaluate_arima, pq_combinations):
            results.append(result)

    best_aic, best_pq = min(results, key=lambda x: x[0])

    print(f"\n--- FINAL RESULT ---")
    print(f"Best model: ARMA{best_pq} with AIC: {best_aic:.4f}")

    best_model = sm.tsa.ARIMA(time_series, order=(best_pq[0], 0, best_pq[1])).fit()
    
    fig = plt.figure(figsize=(14, 8))
    plt.plot(time_series, label='Original Series', color='black', alpha=0.6)
    plt.plot(best_model.fittedvalues, label=f'Best ARMA{best_pq}', color='red', linewidth=2)
    plt.title(f'Best ARMA Model (p={best_pq[0]}, q={best_pq[1]})')
    plt.legend()
    plt.grid(True)
    
    tool.save_figure(fig, 'ex4-arma-optimization', lab_name=LAB_NAME)
    plt.show()

    print(best_model.summary())
    

def run():
    solve2()
    solve3()
    solve4()

if __name__ == "__main__":
    run()