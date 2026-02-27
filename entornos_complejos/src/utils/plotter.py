import numpy as np
from typing import Dict
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
    Ploteamos la nueva métrica f(t) = len(episodio_t)
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

def plot_win_rate(stats: Dict[str, list], window_size: int = 1000, title: str = "Tasa de Éxito (Win Rate)"):
    """
    Calcula y grafica la tasa de éxito (porcentaje de victorias).
    En Blackjack, una recompensa de 1.0 significa victoria.
    """
    rewards = np.array(stats["episode_rewards"])
    # Convertimos recompensas a booleanos (1 si ganó, 0 si perdió o empató)
    wins = (rewards == 1.0).astype(int)
    
    win_rate = moving_average(wins, window_size) * 100 # En porcentaje
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(window_size - 1, len(wins)), win_rate, 
             color='green', linewidth=2, label=f'Win Rate (ventana={window_size})')
    
    plt.title(title)
    plt.xlabel("Episodios")
    plt.ylabel("Tasa de Victorias (%)")
    plt.axhline(y=42.5, color='r', linestyle='--', alpha=0.5, label='Máximo teórico del jugador (Política Óptima)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_multiple_seeds_rewards(data_dict: Dict[str, np.ndarray], window_size: int = 50, title: str = "Rendimiento promedio y varianza sobre varias ejecuciones"):
    """
    Grafica la recompensa media y la varianza sobre múltiples semillas.
    Evalúa el rendimiento promedio sobre múltiples ejecuciones independientes y la varianza del retorno.
    
    Args:
        data_dict: Diccionario donde la clave es el nombre del algoritmo (ej. 'SARSA') 
                   y el valor es una matriz 2D de numpy (semillas x episodios).
        window_size: Tamaño de la ventana para la media móvil.
        title: Título de la gráfica.
    """
    plt.figure(figsize=(12, 6))
    
    # Paleta de colores para diferenciar algoritmos
    colores = ['blue', 'red', 'green', 'purple'] 
    
    for idx, (algo_name, matrix_data) in enumerate(data_dict.items()):
        color = colores[idx % len(colores)]
        
        # Dimensiones: número de semillas y número de episodios
        n_seeds, n_episodes = matrix_data.shape
        
        # Aplicamos la media móvil a la ejecución de cada semilla individualmente
        smoothed_data = []
        for i in range(n_seeds):
            smoothed_data.append(moving_average(matrix_data[i], window_size))
        smoothed_data = np.array(smoothed_data)
        
        # Calculamos la media y la desviación estándar a lo largo del eje de las semillas (axis=0)
        mean_rewards = np.mean(smoothed_data, axis=0)
        std_rewards = np.std(smoothed_data, axis=0)
        
        # Ajustamos el eje X por la pérdida de episodios iniciales debido a la media móvil
        x = np.arange(window_size - 1, n_episodes)
        
        # Graficamos la línea de la media central
        plt.plot(x, mean_rewards, label=f'{algo_name} (Media)', color=color, linewidth=2)
        
        # Sombreamos el área de la varianza (Media ± 1 Desviación Estándar)
        plt.fill_between(x, 
                         mean_rewards - std_rewards, 
                         mean_rewards + std_rewards, 
                         color=color, alpha=0.2, label=f'{algo_name} (±1 StdDev)')
        
    plt.title(title)
    plt.xlabel(f"Episodios (Media Móvil n={window_size})")
    plt.ylabel("Recompensa Promedio")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_multiple_seeds_rewards2(data_dict: Dict[str, np.ndarray], window_size: int = 50, title: str = "Longitud episodio y varianza sobre varias ejecuciones"):
    """
    Grafica la longitud promedio y la varianza sobre múltiples semillas.
    
    Args:
        data_dict: Diccionario donde la clave es el nombre del algoritmo (ej. 'SARSA') 
                   y el valor es una matriz 2D de numpy (semillas x episodios).
        window_size: Tamaño de la ventana para la media móvil.
        title: Título de la gráfica.
    """
    plt.figure(figsize=(12, 6))
    
    # Paleta de colores para diferenciar algoritmos
    colores = ['blue', 'red', 'green', 'purple'] 
    
    for idx, (algo_name, matrix_data) in enumerate(data_dict.items()):
        color = colores[idx % len(colores)]
        
        # Dimensiones: número de semillas y número de episodios
        n_seeds, n_episodes = matrix_data.shape
        
        # Aplicamos la media móvil a la ejecución de cada semilla individualmente
        smoothed_data = []
        for i in range(n_seeds):
            smoothed_data.append(moving_average(matrix_data[i], window_size))
        smoothed_data = np.array(smoothed_data)
        
        # Calculamos la media y la desviación estándar a lo largo del eje de las semillas (axis=0)
        mean_rewards = np.mean(smoothed_data, axis=0)
        std_rewards = np.std(smoothed_data, axis=0)
        
        # Ajustamos el eje X por la pérdida de episodios iniciales debido a la media móvil
        x = np.arange(window_size - 1, n_episodes)
        
        # Graficamos la línea de la media central
        plt.plot(x, mean_rewards, label=f'{algo_name} (Media)', color=color, linewidth=2)
        
        # Sombreamos el área de la varianza (Media ± 1 Desviación Estándar)
        plt.fill_between(x, 
                         mean_rewards - std_rewards, 
                         mean_rewards + std_rewards, 
                         color=color, alpha=0.2, label=f'{algo_name} (±1 StdDev)')
        
    plt.title(title)
    plt.xlabel(f"Episodios (Media Móvil n={window_size})")
    plt.ylabel("Longitud del episodio (paso)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_value_estimation_error(stats_dict: Dict[str, np.ndarray], window_size: int = 50, title: str = "Error Cuadrático Medio en Estimación de Valor (Loss)"):
    """Grafica la evolución del TD Error."""
    plt.figure(figsize=(12, 6))
    colores = ['purple', 'orange', 'cyan', 'brown'] 
    
    for idx, (algo_name, matrix_data) in enumerate(stats_dict.items()):
        color = colores[idx % len(colores)]
        n_seeds, n_episodes = matrix_data.shape
        
        smoothed_data = np.array([moving_average(matrix_data[i], window_size) for i in range(n_seeds)])
        mean_losses = np.mean(smoothed_data, axis=0)
        std_losses = np.std(smoothed_data, axis=0)
        x = np.arange(window_size - 1, n_episodes)
        
        plt.plot(x, mean_losses, label=f'{algo_name} (Media)', color=color, linewidth=2)
        plt.fill_between(x, mean_losses - std_losses, mean_losses + std_losses, color=color, alpha=0.2)
        
    plt.title(title)
    plt.xlabel(f"Episodios (Media Móvil n={window_size})")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
