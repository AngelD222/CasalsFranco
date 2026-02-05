"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the Binomial distribution arm.
"""

import numpy as np
from arms import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución Binomial B(n, p).

        :param n: Número de ensayos (o tamaño del lote). Debe ser entero > 0.
        :param p: Probabilidad de éxito en cada ensayo. Debe estar en [0, 1].
        """
        assert isinstance(n, int) and n > 0, "El número de ensayos n debe ser un entero positivo."
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."
        
        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Binomial.
        Representa el número de éxitos en n ensayos.

        :return: Recompensa obtenida (0 a n).
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.
        E[X] = n * p

        :return: Valor esperado.
        """
        return self.n * self.p

    def __str__(self):
        return f"ArmBinomial(n={self.n}, p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10):
        """
        Genera k brazos binomiales con el mismo n pero diferentes probabilidades p.

        :param k: Número de brazos a generar.
        :param n: Número de ensayos (por defecto 10, según ejemplos típicos).
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(0.05, 0.95)
            p = round(p, 4)
            p_values.add(p)
        
        p_values = list(p_values)
        # Todos los brazos comparten n, pero tienen distinto p
        arms = [ArmBinomial(n, p) for p in p_values]
        
        return arms
