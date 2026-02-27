import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from src.agents import Agent
from src.policies import epsilon_greedy
from src.networks.q_network import SimpleQNetwork

class AgentSarsaSemiGradient2(Agent):
    """
    Agente SARSA Episódico Semi-gradiente usando PyTorch.
    Versión corregida: Incluye normalización de estados y 
    manejo riguroso de estados terminales vs truncados.
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        
        # Hiperparámetros (LR ajustado a 0.0005 por defecto)
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon', 1.0) 
        self.lr = self.hyperparameters.get('lr', 0.0005)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar la Red Neuronal (Aproximador)
        self.q_network = SimpleQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # Caché para SARSA
        self.next_action = None

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normaliza la posición y la velocidad al rango [-1, 1] aproximadamente."""
        # Límites del entorno MountainCar-v0
        low = np.array([-1.2, -0.07])
        high = np.array([0.6, 0.07])
        
        # Escalado Min-Max normalizado entre -1 y 1
        state_norm = 2.0 * (state - low) / (high - low) - 1.0
        return state_norm

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Paso auxiliar para obtener Q-values normalizando el estado previamente."""
        state_norm = self.normalize_state(state)
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]

    def _choose_action_internal(self, state: np.ndarray) -> int:
        """Aplica la política epsilon-greedy sobre los Q-values aproximados."""
        q_values = self._get_q_values(state)
        return epsilon_greedy(q_values, self.epsilon, self.action_dim)

    def get_action(self, state: np.ndarray) -> int:
        """Obtiene la acción actual, usando la caché si corresponde (SARSA on-policy)."""
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
            
        return self._choose_action_internal(state)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Realiza el paso de actualización por gradiente descendente.
        Corrige el sesgo de truncamiento continuando el bootstrapping.
        """
        # Normalizar estados
        obs_norm = self.normalize_state(obs)
        next_obs_norm = self.normalize_state(next_obs)
        
        # Convertir a tensores
        obs_tensor = torch.FloatTensor(obs_norm).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs_norm).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # 1. Calcular el valor Q actual: Q(S, A, w)
        q_values = self.q_network(obs_tensor)
        q_current = q_values[action]
        
        # 2. Calcular el Target (Semi-gradiente)
        if terminated:
            # SOLO si es un estado terminal VERDADERO (llegó a la bandera), Q(S', A') es 0
            target = reward_tensor
        else:
            # Si no ha llegado a la meta (incluso si se truncó por límite de tiempo), hay bootstrapping
            next_action_internal = self._choose_action_internal(next_obs)
            self.next_action = next_action_internal # Guardar en caché
            
            # Obtener Q(S', A')
            with torch.no_grad():
                q_next_values = self.q_network(next_obs_tensor)
                q_next = q_next_values[next_action_internal]
                
            target = reward_tensor + self.gamma * q_next

        # 3. Optimización
        loss = self.loss_fn(q_current, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
