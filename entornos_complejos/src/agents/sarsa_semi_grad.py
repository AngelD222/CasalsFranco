import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from src.agents import Agent
from src.policies import epsilon_greedy
from src.networks.q_network import SimpleQNetwork

class AgentSarsaSemiGradient(Agent):
    """
    Agente SARSA Episódico Semi-gradiente usando PyTorch.
    Aproxima la función Q(S,A,w) usando una red neuronal.
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        
        # Hiperparámetros
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon', 1.0) 
        self.lr = self.hyperparameters.get('lr', 0.001)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar la Red Neuronal (Aproximador)
        self.q_network = SimpleQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # Inyección dinámica para la métrica del Error
        self.current_episode_losses = []
        if "episode_losses" not in self.training_stats:
            self.training_stats["episode_losses"] = []
        
        # Caché para SARSA (necesitamos la siguiente acción A_{t+1})
        self.next_action = None

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Paso auxiliar para obtener Q-values en formato numpy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]

    def _choose_action_internal(self, state: np.ndarray) -> int:
        """Aplica la política epsilon-greedy sobre los Q-values aproximados."""
        q_values = self._get_q_values(state)
        return epsilon_greedy(q_values, self.epsilon, self.action_dim)

    def get_action(self, state: np.ndarray) -> int:
        """
        En SARSA, la acción A_{t+1} ya fue elegida en el update del paso anterior.
        Si la tenemos en caché, la usamos. Si no, elegimos una nueva.
        """
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
            
        return self._choose_action_internal(state)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Realiza el paso de actualización por gradiente descendente.
        Fórmula: w <- w + alpha * [R + gamma * Q(S', A', w) - Q(S, A, w)] * grad(Q)
        """
        # Convertir a tensores
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # 1. Calcular el valor Q actual: Q(S, A, w)
        q_values = self.q_network(obs_tensor)
        q_current = q_values[action] # Solo nos interesa el Q de la acción tomada
        
        # 2. Calcular el Target (Semi-gradiente)
        if terminated or truncated:
            # Si es estado terminal, Q(S', A') es 0
            target = reward_tensor
        else:
            # Elegir A_{t+1} con la política actual para SARSA On-Policy
            next_action_internal = self._choose_action_internal(next_obs)
            self.next_action = next_action_internal # Guardar en caché
            
            # Obtener Q(S', A')
            with torch.no_grad(): # DETACH: Esto es lo que lo hace "Semi-gradiente"
                q_next_values = self.q_network(next_obs_tensor)
                q_next = q_next_values[next_action_internal]
                
            target = reward_tensor + self.gamma * q_next

        # 3. Optimización
        loss = self.loss_fn(q_current, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
