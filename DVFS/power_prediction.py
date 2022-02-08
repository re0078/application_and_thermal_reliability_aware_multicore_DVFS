import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class PowerPredictor:
    predictor: SGDRegressor
    scaler: StandardScaler = StandardScaler()
    z: int

    def __init__(self, z=8, lr=1e-3):
        self.z = z
        self.predictor = SGDRegressor(loss="log", learning_rate=lr)

    def fill_power_history(self, power_hist):
        hist_size = power_hist.shape[0]
        if hist_size < self.z:
            power_hist = np.concatenate((np.zeros((self.z - hist_size, power_hist.shape[1])), power_hist))
        return power_hist

    def partial_fit(self, power_hist: np.ndarray, next_power_emp):
        power_hist = self.fill_power_history(power_hist)

        self.scaler = self.scaler.partial_fit(power_hist)
        power_hist = self.scaler.transform(power_hist)
        self.predictor.partial_fit(power_hist, next_power_emp)

    def predict(self, power_hist: np.ndarray):
        power_hist = self.fill_power_history(power_hist)
        power_hist = self.scaler.transform(power_hist)
        return self.predictor.predict(power_hist)
