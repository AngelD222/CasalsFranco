import os
import gc
import torch
import numpy as np
import gymnasium as gym

def set_seed(seed: int = 2024):
    """
    Fija todas las semillas de aleatoriedad para garantizar la reproducibilidad,
    tal y como recomienda el profesor en el Anexo 5.4.
    """
    # Configuración del dispositivo (CPU / GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Liberación de memoria para evitar problemas de consumo en GPU
    gc.collect() # Ejecuta el recolector de basura de Python
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Vacía la caché de memoria en GPU
        
    # Depuración de errores en CUDA
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Muestra errores de CUDA en el punto exacto
    
    # Fijar la semilla en Python
    os.environ['PYTHONHASHSEED'] = str(seed) # Evita variabilidad en hashing de Python
    
    # Fijar la semilla en NumPy
    np.random.seed(seed) 
    
    # Fijar la semilla en PyTorch
    torch.manual_seed(seed) 
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True # Hace las operaciones de CUDNN determinísticas
        torch.backends.cudnn.benchmark = False # Desactiva optimizaciones para evitar variabilidad

def make_env(env_name: str, seed: int = 2024):
    """
    Crea un entorno de Gymnasium y le inyecta la semilla inicial.
    """
    env = gym.make(env_name)
    # Establece la semilla en el entorno de Gymnasium (corregido el error tipográfico del PDF)
    env.reset(seed=seed) 
    return env
