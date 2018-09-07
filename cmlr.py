import numpy as np

def fit_cmlr(log_ml_a, color_a):
    p, cov = np.polyfit(color_a, log_ml_a, deg=1, cov=True)
    return p, cov
