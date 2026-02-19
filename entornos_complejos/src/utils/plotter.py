import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def moving_average(data: List[float], window_size: int = 50) -> np.ndarray:
    """
    Calcula la media móvil para suavizar las gráficas ruidosas de RL.
    La media móvil promedia los resultados de múltiples episodios recientes para mostrar una tendencia clara de si el agente está mejorando
    """
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_episode_rewards(stats: Dict[str, list], window_size: int = 50, title: str = "Recompensa por Episodio"):
    """
    Dibuja la evolución de las recompensas a lo largo del entrenamiento.
    """
    rewards = stats["episode_rewards"]
    smoothed_rewards = moving_average(rewards, window_size)
    
    plt.figure(figsize=(10, 5))
    # Gráfica cruda en fondo claro
    plt.plot(rewards, alpha=0.3, color='blue', label='Recompensa cruda')
    # Gráfica suavizada (la importante para el análisis)
    plt.plot(np.arange(window_size - 1, len(rewards)), smoothed_rewards, 
             color='darkblue', linewidth=2, label=f'Media móvil (n={window_size})')
    
    plt.title(title)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_episode_lengths(stats: Dict[str, list], window_size: int = 50, title: str = "Longitud del Episodio (f(t))"):
    """
    Dibuja la métrica f(t) = len(episodio_t) exigida en la guía de la práctica.
    """
    lengths = stats["episode_lengths"]
    smoothed_lengths = moving_average(lengths, window_size)
    
    plt.figure(figsize=(10, 5))
    plt.plot(lengths, alpha=0.3, color='orange', label='Longitud cruda')
    plt.plot(np.arange(window_size - 1, len(lengths)), smoothed_lengths, 
             color='darkorange', linewidth=2, label=f'Media móvil (n={window_size})')
    
    plt.title(title)
    plt.xlabel("Episodios")
    plt.ylabel("Pasos por Episodio")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
