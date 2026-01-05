import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso

from signal_processing_core import SignalToolkit

LAB_NAME = 'Lab10'
tool = SignalToolkit(output_dir='charts')


def generate_time_series_data():
    trend, season, noise = tool.generate_time_series()
    time_series = trend + season + noise
    return time_series, trend, season, noise


def ar_model(ts, p):
    # 2
    train_data = ts[:-p]
    test_data = ts[-p:]
    y = train_data[p:]
    m = len(y)

    Y = np.zeros((m, p))
    for i in range(p):
        Y[:, i] = train_data[i:i+m]

    big_gamma = Y.T @ Y 
    small_gamma = Y.T @ y
    x_star = np.linalg.inv(big_gamma) @ small_gamma

    pred = []
    for i in range(p):
        last_p = np.append(train_data, pred)[-p:]
        pred_val = x_star @ last_p
        pred.append(pred_val)

    return train_data, test_data, np.array(pred), x_star


def ar_model_greedy(ts, p):
    # 3a
    train_data = ts[:-p]
    test_data = ts[-p:]
    y = train_data[p:]
    m = len(y)

    Y = np.zeros((m, p))
    for i in range(p):
        Y[:, i] = train_data[i:i+m]

    selected_features = []
    x_sparse = np.zeros(p)
    
    for step in range(p):
        best_error = float('inf')
        best_feature = None
        
        for feature in range(p):
            if feature in selected_features:
                continue
            
            test_features = selected_features + [feature]
            Y_subset = Y[:, test_features]
            
            x_subset = np.linalg.lstsq(Y_subset, y, rcond=None)[0]
            
            error = np.sum((y - Y_subset @ x_subset) ** 2)
            
            if error < best_error:
                best_error = error
                best_feature = feature
                best_x = x_subset
        
        if best_feature is not None:
            selected_features.append(best_feature)
            
        if best_error < 1e-6 or len(selected_features) >= p // 2:
            break
    
    for i, idx in enumerate(selected_features):
        x_sparse[idx] = best_x[i]

    pred = []
    for i in range(p):
        last_p = np.append(train_data, pred)[-p:]
        pred_val = x_sparse @ last_p
        pred.append(pred_val)

    return train_data, test_data, np.array(pred), x_sparse


def ar_model_l1(ts, p, alpha=0.1):
    # 3b
    train_data = ts[:-p]
    test_data = ts[-p:]
    y = train_data[p:]
    m = len(y)

    Y = np.zeros((m, p))
    for i in range(p):
        Y[:, i] = train_data[i:i+m]

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(Y, y)
    x_sparse = lasso.coef_

    pred = []
    for i in range(p):
        last_p = np.append(train_data, pred)[-p:]
        pred_val = x_sparse @ last_p
        pred.append(pred_val)

    return train_data, test_data, np.array(pred), x_sparse


def polynomial_roots_companion(coefficients):
    # 4
    p = len(coefficients)
    
    if p == 0:
        return np.array([])
    
    if p == 1:
        return np.array([1.0 / coefficients[0]]) if coefficients[0] != 0 else np.array([])
    
    companion = np.zeros((p, p))
    companion[0, :] = coefficients
    companion[1:, :-1] = np.eye(p - 1)
    
    roots = np.linalg.eigvals(companion)
    
    return roots


def check_stationarity(coefficients):
    # 5
    roots = polynomial_roots_companion(coefficients)
    
    if len(roots) == 0:
        return True, roots, np.array([])
    
    magnitudes = np.abs(roots)
    is_stationary = np.all(magnitudes > 1.0)
    
    return is_stationary, roots, magnitudes


def visualize_results(time_series):
    fig1, axs1 = plt.subplots(4, 1, figsize=(12, 10))
    p_values = [2, 50, 75, 125]
    
    for idx, p in enumerate(p_values):
        train_data, test_data, pred, coef = ar_model(time_series, p)
        x_pred = np.arange(len(train_data), len(train_data) + len(pred))
        
        axs1[idx].plot(train_data, label=f"Train data", alpha=0.7)
        axs1[idx].plot(x_pred, pred, "r-", label=f"AR Prediction (p={p})", linewidth=2)
        axs1[idx].plot(x_pred, test_data, "g--", label=f"Actual values", linewidth=2)
        axs1[idx].set_title(f"Standard AR Model (p={p})")
        axs1[idx].legend()
        axs1[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    tool.save_figure(fig1, 'ex2_standard_ar', lab_name=LAB_NAME)
    
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 10))
    p = 75
    
    train_data, test_data, pred_std, coef_std = ar_model(time_series, p)
    x_pred = np.arange(len(train_data), len(train_data) + len(pred_std))
    
    axs2[0].plot(train_data, label="Train data", alpha=0.7)
    axs2[0].plot(x_pred, pred_std, "r-", label="Standard AR", linewidth=2)
    axs2[0].plot(x_pred, test_data, "g--", label="Actual", linewidth=2)
    axs2[0].set_title(f"Standard AR Model (p={p})")
    axs2[0].legend()
    axs2[0].grid(True, alpha=0.3)
    
    train_data, test_data, pred_greedy, coef_greedy = ar_model_greedy(time_series, p)
    
    axs2[1].plot(train_data, label="Train data", alpha=0.7)
    axs2[1].plot(x_pred, pred_greedy, "r-", label=f"Greedy Sparse AR ({np.sum(coef_greedy != 0)} non-zero)", linewidth=2)
    axs2[1].plot(x_pred, test_data, "g--", label="Actual", linewidth=2)
    axs2[1].set_title(f"Greedy Sparse AR Model (p={p})")
    axs2[1].legend()
    axs2[1].grid(True, alpha=0.3)
    
    train_data, test_data, pred_l1, coef_l1 = ar_model_l1(time_series, p, alpha=0.5)
    
    axs2[2].plot(train_data, label="Train data", alpha=0.7)
    axs2[2].plot(x_pred, pred_l1, "r-", label=f"L1 Regularized AR ({np.sum(np.abs(coef_l1) > 1e-6)} non-zero)", linewidth=2)
    axs2[2].plot(x_pred, test_data, "g--", label="Actual", linewidth=2)
    axs2[2].set_title(f"L1 Regularized AR Model (p={p})")
    axs2[2].legend()
    axs2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    tool.save_figure(fig2, 'ex3_sparse_ar_comparison', lab_name=LAB_NAME)
    
    fig3, axs3 = plt.subplots(3, 1, figsize=(12, 8))
    
    axs3[0].stem(coef_std, label="Standard AR")
    axs3[0].set_title("Standard AR Coefficients")
    axs3[0].grid(True, alpha=0.3)
    
    axs3[1].stem(coef_greedy, label="Greedy Sparse", linefmt='r-', markerfmt='ro')
    axs3[1].set_title(f"Greedy Sparse Coefficients ({np.sum(coef_greedy != 0)} non-zero)")
    axs3[1].grid(True, alpha=0.3)
    
    axs3[2].stem(coef_l1, label="L1 Regularized", linefmt='g-', markerfmt='go')
    axs3[2].set_title(f"L1 Regularized Coefficients ({np.sum(np.abs(coef_l1) > 1e-6)} non-zero)")
    axs3[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    tool.save_figure(fig3, 'ex3_coefficient_comparison', lab_name=LAB_NAME)
    
    return coef_std, coef_greedy, coef_l1


def analyze_stationarity(coef_std, coef_greedy, coef_l1):
    models = [
        ("Standard AR", coef_std),
        ("Greedy Sparse AR", coef_greedy),
        ("L1 Regularized AR", coef_l1)
    ]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    for idx, (name, coef) in enumerate(models):
        is_stationary, roots, magnitudes = check_stationarity(coef)
        
        print(f"\n{name}:")
        print(f"Number of coefficients: {len(coef)}")
        print(f"Non-zero coefficients: {np.sum(np.abs(coef) > 1e-6)}")
        print(f"Is stationary: {is_stationary}")
        print(f"Root magnitudes: min={magnitudes.min():.4f}, max={magnitudes.max():.4f}")
        
        if not is_stationary:
            inside_unit = magnitudes <= 1.0
            print(f"  WARNING: {np.sum(inside_unit)} roots inside/on unit circle!")
        
        angles = np.angle(roots)
        axs[idx].scatter(angles, magnitudes, c='red', s=50, alpha=0.6, label='Roots')
        
        theta = np.linspace(0, 2*np.pi, 100)
        axs[idx].plot(theta, np.ones_like(theta), 'b--', linewidth=2, label='Unit Circle')
        
        axs[idx].set_title(f"{name}\n{'Stationary' if is_stationary else 'Non-stationary'}", 
                          fontsize=10)
        axs[idx].legend(loc='upper right', fontsize=8)
        axs[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    tool.save_figure(fig, 'ex5_stationarity_analysis', lab_name=LAB_NAME)


def run():
    time_series, trend, season, noise = generate_time_series_data()
    coef_std, coef_greedy, coef_l1 = visualize_results(time_series)
    analyze_stationarity(coef_std, coef_greedy, coef_l1)

if __name__ == "__main__":
    run()