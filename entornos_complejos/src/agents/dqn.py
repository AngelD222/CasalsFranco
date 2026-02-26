import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.agents import Agent
from src.policies import epsilon_greedy
from src.networks.q_network import SimpleQNetwork
from src.utils.replay_buffer import ReplayBuffer

class AgentDQN(Agent):
    """
    Agente Deep Q-Network (DQN) con Experience Replay y Target Network.
    Algoritmo de control Off-Policy.
    """
    def __init__(self, env, hyperparameters):
        super().__init__(env, hyperparameters)
        
        # Hiperparámetros
        self.gamma = self.hyperparameters.get('gamma', 0.99)
        self.epsilon = self.hyperparameters.get('epsilon', 1.0)
        self.lr = self.hyperparameters.get('lr', 0.001)
        self.batch_size = self.hyperparameters.get('batch_size', 64)
        self.target_update_freq = self.hyperparameters.get('target_update_freq', 100)
        buffer_capacity = self.hyperparameters.get('buffer_capacity', 10000)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Red Principal (Policy Network) - La que entrenamos
        self.policy_net = SimpleQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 2. Red Objetivo (Target Network) - La que usamos para el target y congelamos
        self.target_net = SimpleQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Modo evaluación (no gradientes)
        
        # Usamos Huber Loss (Smooth L1) en lugar de MSE porque es más robusta frente a outliers
        self.loss_fn = nn.SmoothL1Loss()
        
        # 3. Memoria de Repetición
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Contador de pasos para sincronizar la red objetivo
        self.step_count = 0

    def get_action(self, state: np.ndarray) -> int:
        """Selecciona acción usando epsilon-greedy sobre la Red Principal"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        return epsilon_greedy(q_values, self.epsilon, self.action_dim)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Almacena la transición en memoria y entrena con un mini-batch si hay suficientes datos.
        """
        # 1. Guardar en memoria (Off-policy: aprendemos de experiencias pasadas)
        done = terminated or truncated
        self.memory.push(obs, action, reward, next_obs, done)
        
        self.step_count += 1
        
        # 2. Comprobar si tenemos suficientes datos para un batch
        if len(self.memory) < self.batch_size:
            return
            
        # 3. Muestrear un mini-batch aleatorio de la memoria
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 4. Calcular Q(S, A) actuales con la Red Principal
        # .gather extrae los valores Q correspondientes a las acciones que realmente se tomaron
        current_q = self.policy_net(states).gather(1, actions)
        
        # 5. Calcular los valores Target con la Red Objetivo (Target Network)
        with torch.no_grad():
            # max(1)[0] obtiene el valor máximo por cada fila (batch)
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Si el estado es terminal (done=1), el valor futuro es 0
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        # 6. Optimizar la Red Principal
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping: Evita que los gradientes exploten (buena práctica en DQN)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # 7. Sincronizar la Red Objetivo cada 'target_update_freq' pasos
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
