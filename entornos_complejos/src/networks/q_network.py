import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleQNetwork(nn.Module):
    """
    Aproximador de función lineal/no-lineal simple para Q(s, a, w).
    Diseñado para espacios de estados continuos pequeños.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super(SimpleQNetwork, self).__init__()
        # Red neuronal feed-forward sencilla
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcula los valores Q para todas las acciones dado un estado.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)
