# Importación directa de las clases desde sus respectivos archivos
from .clase_BASE import Agent
from .monte_carlo import AgentMonteCarloTodasVisitas

from monte_carlo_off import AgentMonteCarloOffPolicy
from q_learning import AgentQLearning
from sarsa import AgentSARSA

# Lista de clases públicas accesibles al importar desde src.agents
__all__ = [
    'Agent',
    'AgentMonteCarloTodasVisitas',
    'AgentMonteCarloOffPolicy',
    'AgentQLearning',
    'AgentSARSA'
]
