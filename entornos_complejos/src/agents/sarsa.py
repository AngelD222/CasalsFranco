import numpy as np
from collections import defaultdict
from src.agents.clase_BASE import Agent
from src.policies.epsilon_greedy import epsilon_greedy

class AgentSARSA(Agent):
    """
    Agente SARSA
    Aprende actualizando Q(S,A) en cada paso usando la acción real que tomará en S'.
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon', 0.1)
        self.alpha = self.hyperparameters.get('alpha', 0.1) # Tasa de aprendizaje
        self.action_space_n = self.env.action_space.n
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
        self.next_action_cache = None 

    def get_action(self, state):
        # Si ya calculamos la acción en el update anterior, la usamos
        if self.next_action_cache is not None:
            action = self.next_action_cache
            self.next_action_cache = None
            return action
            
        return epsilon_greedy(self.q_table[state], self.epsilon, self.action_space_n)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        done = terminated or truncated
        
        if not done:
            # Calculamos y cacheamos la siguiente acción A' usando la política actual
            self.next_action_cache = self.get_action(next_obs)
            next_q = self.q_table[next_obs][self.next_action_cache]
        else:
            next_q = 0.0 # En estados terminales, el valor futuro es 0
            
        # Fórmula de actualización SARSA: Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[obs][action]
        
        self.q_table[obs][action] += self.alpha * td_error
