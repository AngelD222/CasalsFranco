# Aquí vamos a definir cómo decide el agente. 
# La política $\epsilon$-greedy elige la mejor acción conocida con probabilidad $1-\epsilon$, y una acción exploratoria al azar con probabilidad $\epsilon$. 
# Además implementamos el decaimiento de Epsilon Decay para que el agente deje de explorar gradualmente conforme se vuelve un experto.

import numpy as np
import random

def epsilon_greedy(q_values_state, epsilon, action_space_n):
    """
    Selecciona una acción usando la política epsilon-greedy.
    
    Args:
        q_values_state (np.array): Los valores Q(s, a) para el estado actual.
        epsilon (float): Probabilidad de exploración [0, 1].
        action_space_n (int): Número total de acciones posibles.
        
    Returns:
        int: La acción seleccionada.
    """
    if random.random() < epsilon:
        # Exploración: elegimos una acción completamente al azar
        return random.randint(0, action_space_n - 1)
    else:
        # Explotación: elegimos la mejor acción (rompiendo empates al azar)
        max_val = np.max(q_values_state)
        best_actions = np.where(q_values_state == max_val)[0]
        return np.random.choice(best_actions)
