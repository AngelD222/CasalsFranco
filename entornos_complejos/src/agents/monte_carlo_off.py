import numpy as np
from collections import defaultdict
from src.agents import Agent

class AgentMonteCarloOffPolicy(Agent):
    """
    Agente Monte Carlo Off-Policy con Muestreo de Importancia Ponderado.
    Aprende una política determinista óptima (pi) mientras sigue una política exploratoria (b).
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        self.gamma = self.hyperparameters.get('gamma', 1.0)
        self.epsilon = self.hyperparameters.get('epsilon', 0.1) 
        self.action_space_n = self.env.action_space.n
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
        # C(s,a) es la suma acumulada de los pesos del muestreo de importancia
        self.C = defaultdict(lambda: np.zeros(self.action_space_n))
        
        self.episode_buffer = []

    def get_action(self, state):
        """
        Política de Comportamiento (b): epsilon-greedy
        Genera el comportamiento exploratorio.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_n)
        else:
            # Si hay empate en Q, argmax devuelve el primero. Añadir desempate aleatorio.
            return np.argmax(self.q_table[state])

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        self.episode_buffer.append((obs, action, reward))
        if terminated or truncated:
            self._aprender_de_episodio()
            self.episode_buffer = []

    def _aprender_de_episodio(self):
        G = 0.0
        W = 1.0 # Peso inicial del muestreo de importancia
        
        for obs, action, reward in reversed(self.episode_buffer):
            G = reward + self.gamma * G
            
            # Actualizamos C(s,a)
            self.C[obs][action] += W
            
            # Actualizamos Q(s,a) usando el Muestreo de Importancia Ponderado
            error = G - self.q_table[obs][action]
            self.q_table[obs][action] += (W / self.C[obs][action]) * error
            
            # ¿Cuál es la acción que tomaría nuestra política objetivo (pi)?
            best_action = np.argmax(self.q_table[obs])
            
            # Si la acción que tomamos (b) no coincide con la que tomaría (pi), 
            # la probabilidad bajo pi es 0, por lo que W se vuelve 0 y el episodio ya no aporta más hacia atrás.
            if action != best_action:
                break
                
            # Actualizamos W dividiendo por la probabilidad de la política de comportamiento
            # Probabilidad epsilon-greedy: (1 - epsilon) + (epsilon / |A|) si es la mejor, o (epsilon / |A|) si no lo es.
            # Como no hemos hecho break, sabemos que action == best_action
            prob_b = (1.0 - self.epsilon) + (self.epsilon / self.action_space_n)
            W = W * (1.0 / prob_b)
