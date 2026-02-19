import numpy as np

def get_epsilon_decay(episode, initial_eps=1.0, final_eps=0.01, decay_rate=0.005):
    """
    Calcula el valor de epsilon con un decaimiento exponencial.
    Formula: eps = final_eps + (initial_eps - final_eps) * exp(-decay_rate * episode)
    
    Args:
        episode (int): Número del episodio actual.
        initial_eps (float): Valor inicial de epsilon (suele ser 1.0 para 100% exploración).
        final_eps (float): Valor mínimo al que llegará epsilon (nunca deja de explorar del todo).
        decay_rate (float): Velocidad a la que decae la exploración.
        
    Returns:
        float: El valor de epsilon para el episodio actual.
    """
    return final_eps + (initial_eps - final_eps) * np.exp(-decay_rate * episode)
