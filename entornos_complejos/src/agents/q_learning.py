import numpy as np
from collections import defaultdict
from src.agents.clase_BASE import Agent
from src.policies.epsilon_greedy import epsilon_greedy

class AgentQLearning(Agent):
    """
    Agente Q-Learning (Off-Policy).
    Aprende actualizando Q(S,A) usando el máximo valor posible en S', ignorando la política.
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon', 0.1)
        self.alpha = self.hyperparameters.get('alpha', 0.1)
        self.action_space_n = self.env.action_space.n
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))

    def get_action(self, state):
        return epsilon_greedy(self.q_table[state], self.epsilon, self.action_space_n)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        done = terminated or truncated
        
        if not done:
            # Seleccionamos el MEJOR valor futuro posible (Off-Policy), sin importar qué haremos
            best_next_q = np.max(self.q_table[next_obs])
        else:
            best_next_q = 0.0
            
        # Fórmula de actualización Q-Learning: Q(S,A) <- Q(S,A) + alpha * [R + gamma * max(Q(S',a)) - Q(S,A)]
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[obs][action]
        
        self.q_table[obs][action] += self.alpha * td_error
