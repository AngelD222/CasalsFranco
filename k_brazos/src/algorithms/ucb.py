"""
Module: algorithms/ucb.py
Description: Implementación del algoritmo UCB1 (Upper Confidence Bound).
"""

import numpy as np
from algorithms.algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = np.sqrt(2)):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de confianza (exploración). 
                  normalmente c=sqrt(2) para cotas de Hoeffding, pero ajustable.
        """
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona el brazo que maximiza el límite superior de confianza (UCB).
        """
        # 1. Inicialización: Asegurar que cada brazo se ha probado al menos una vez
        # para evitar división por cero en el término de exploración.
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # 2. Cálculo de UCB
        # t es el paso de tiempo total actual (suma de todas las selecciones)
        t = np.sum(self.counts)
        
        # Fórmula UCB: Q(a) + c * sqrt(ln(t) / N(a))
        # Usamos array operations de numpy para eficiencia
        exploration_term = self.c * np.sqrt(np.log(t) / self.counts)
        ucb_values = self.values + exploration_term
        
        # 3. Selección: Brazo con mayor valor UCB
        # En caso de empate, rompemos la simetría aleatoriamente
        max_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_value)[0]
        chosen_arm = np.random.choice(best_arms)
        
        return chosen_arm
