# Vamos a guardar el dato de la función f(t) = len(episodio_t)

from tqdm import tqdm
from src.policies import epsilon_decay

def train_agent(env, agent, n_episodes, initial_eps=1.0, final_eps=0.01, decay_rate=0.005):
    """
    Función genérica para entrenar cualquier agente en cualquier entorno de Gymnasium.
    Implementa el decaimiento de epsilon y recopila métricas de rendimiento.
    """
    for episode in tqdm(range(n_episodes), desc="Entrenando agente"):
        # 1. Actualizamos el valor de epsilon para este episodio
        current_epsilon = epsilon_decay(episode, initial_eps, final_eps, decay_rate)
        # Inyectamos el epsilon actualizado al agente (ya que nuestra clase base lo usará)
        agent.epsilon = current_epsilon 
        
        # 2. Reiniciamos el entorno
        obs, info = env.reset()
        done = False
        
        # Variables para las métricas del episodio actual
        episode_reward = 0.0
        episode_length = 0  # Esta es la métrica f(t) que pide el profesor
        
        # 3. Jugamos un episodio completo
        while not done:
            # El agente decide qué hacer
            action = agent.get_action(obs)
            
            # El entorno reacciona a la acción
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # El agente aprende de la experiencia
            agent.update(obs, action, next_obs, reward, terminated, truncated, info)
            
            # Actualizamos contadores
            episode_reward += reward
            episode_length += 1
            
            # Comprobamos si el episodio ha terminado
            done = terminated or truncated
            obs = next_obs
            
        # 4. Guardamos las estadísticas del episodio en el diccionario del agente
        agent.training_stats["episode_rewards"].append(episode_reward)
        agent.training_stats["episode_lengths"].append(episode_length)
        agent.training_stats["epsilon_history"].append(current_epsilon)
        
    return agent.training_stats
