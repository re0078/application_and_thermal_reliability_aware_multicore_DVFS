import numpy as np
from scipy.constants import Boltzmann
from scipy.special import gamma


class ApplicationReliability:
    tf_lambda_zero = 1e-6
    tf_delta = 1
    tf_V_max = 10  # todo change

    def get_tf_lambda(self, V):
        return self.tf_lambda_zero * 10 ** ((self.tf_V_max - V) / self.tf_delta)

    def get_FR(self, w, V, f, FVI=1):
        return np.exp(-self.get_tf_lambda(V) * FVI * w / f)


class ThermalReliability:
    MATERIAL_BASED_CONST = 1.1
    weibull_slope = 2
    ACTIVATION_ENERGY_CONST = 0.9

    def get_const(self):
        beta = self.weibull_slope
        n = self.MATERIAL_BASED_CONST
        J = 1  # TODO
        return np.power((1 / (gamma(1 + 1 / beta) * (J ** -n))), beta)

    def get_TR(self, t, T):
        beta = self.weibull_slope
        k = Boltzmann
        return np.exp(-self.get_const() * (t ** beta) * np.sum(np.exp(-self.ACTIVATION_ENERGY_CONST * beta / (k * T))))


class PowerModel:
    I0 = 1  # TODO
    eta = 1  # TODO
    thermal_voltage = 1  # TODO
    threshold_voltage = 1  # TODO
    switching_activity_factor = 1  # TODO
    avg_capacitance = 1  #

    def get_power(self, V, f):
        p_static = self.I0 * np.exp(-self.threshold_voltage / (self.eta * self.thermal_voltage))
        p_dynamic = self.switching_activity_factor * self.avg_capacitance * np.power(V, 2) * f
        return p_static + p_dynamic
