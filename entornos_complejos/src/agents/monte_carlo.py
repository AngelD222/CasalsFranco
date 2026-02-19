import numpy as np
from collections import defaultdict
from src.agents import Agent
from src.policies import epsilon_greedy

class AgentMonteCarloTodasVisitas(Agent):
    """
    Agente que implementa Monte Carlo On-Policy
    Aprende calculando el retorno G_t hacia atrás al final de cada episodio
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        
        # Hiperparámetros
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon', 0.1) 
        self.action_space_n = self.env.action_space.n
        
        # Memoria del agente
        # Usamos defaultdict para inicializar automáticamente estados nuevos con arrays de ceros
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
        
        # Contador de visitas N(s, a)
        self.returns_count = defaultdict(lambda: np.zeros(self.action_space_n))
        
        # Buffer para guardar la trayectoria del episodio actual
        self.episode_buffer = []

    def get_action(self, state):
        """Usa la política epsilon-greedy importada de nuestro módulo"""
        return epsilon_greedy(self.q_table[state], self.epsilon, self.action_space_n)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Guarda la experiencia. Si el episodio termina, procesa el buffer
        calculando el retorno hacia atrás y actualiza la tabla Q.
        """
        # 1. Guardamos la transición en la memoria a corto plazo
        self.episode_buffer.append((obs, action, reward))
        
        # 2. Solo aprendemos si el episodio ha terminado
        if terminated or truncated:
            self._aprender_de_episodio()
            
            # Limpiamos el buffer para el siguiente episodio
            self.episode_buffer = []

    def _aprender_de_episodio(self):
        """Método interno para calcular G_t hacia atrás"""
        G = 0
        # Recorremos el episodio desde el último paso hasta el primero
        for obs, action, reward in reversed(self.episode_buffer):
            # Fórmula del retorno acumulado con descuento
            G = reward + self.gamma * G
            
            # Actualización de todas las visitas 
            self.returns_count[obs][action] += 1
            N = self.returns_count[obs][action]
            
            # Actualización incremental del valor Q
            error = G - self.q_table[obs][action]
            self.q_table[obs][action] += (1.0 / N) * error
