# El agente debe inicializar lo necesario para el aprendizaje en __init__ , tener una función get_action para su política  y una función update para aplicar el algoritmo
# Además hacemos una llamada final a agent.stats() para obtener los resultados

import gymnasium as gym
from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    """
    Clase base abstracta para todos los agentes de Reinforcement Learning.
    """
    def __init__(self, env: gym.Env, hyperparameters: Dict[str, Any]):
        """Inicializa todo lo necesario para el aprendizaje"""
        self.env = env
        self.hyperparameters = hyperparameters
        
        # Diccionario para guardar métricas
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [], # métrica f(t) nueva
            "epsilon_history": []
        }

    @abstractmethod
    def get_action(self, state: Any) -> Any:
        """
        Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente.
        """
        pass

    @abstractmethod
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Con la muestra (s, a, s', r) e información complementaria aplicamos el algoritmo
        update() no es más que el algoritmo de aprendizaje del agente
        """
