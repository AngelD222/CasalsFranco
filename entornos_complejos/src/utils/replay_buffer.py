import numpy as np
import random
from collections import deque
from typing import Tuple, List

class ReplayBuffer:
    """
    Memoria de repetición (Experience Replay) para romper la correlación 
    temporal de los datos, estabilizando el entrenamiento de la red neuronal.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Almacena una transición en la memoria."""
        # Aseguramos que los estados sean arrays de numpy para facilitar el paso a tensores luego
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un mini-lote aleatorio de transiciones.
        Devuelve arrays de numpy listos para ser convertidos a tensores de PyTorch.
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Desempaquetamos el batch de tuplas en listas separadas
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32) # Los booleanos se convierten a 1.0 y 0.0
        )
    
    def __len__(self):
        return len(self.buffer)
