import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class DeterministicOptimizer:
    """
    Modelo determinista basado en optimización no lineal con SciPy.
    Utilizado como baseline o alternativa rápida.
    """
    def __init__(self, config: dict):
        self.beta = config.get('beta', 0.01)
        self.D = config.get('demand_scale', 1000)
        self.C = config.get('train_capacity', 300)
        self.T = 1 # Periodo simplificado

    def _objective(self, lam):
        # p(lambda) = -1/beta * log(lambda/D)
        precio = -(1 / self.beta) * np.log(lam / self.D)

        ingresos = precio * lam * self.T
        return -ingresos

    def solve(self, **kwargs):
        """
        Ejecuta la optimización determinista.
        """
        constraints = ({'type': 'ineq', 'fun': lambda lam: self.C - lam})
        bounds = [(1e-6, None)]
        initial_guess = [self.C / 2]

        res = minimize(self._objective, initial_guess, constraints=constraints, bounds=bounds)

        if res.success:
            lambda_opt = res.x[0]
            precio_opt = -(1 / self.beta) * np.log(lambda_opt / self.D)
            return {
                'status': 'OPTIMAL',
                'best_price': float(precio_opt),
                'expected_demand': float(lambda_opt),
                'revenue': float(-res.fun)
            }
        else:
            logger.error("La optimización determinista falló.")
            return {'status': 'FAILED'}
