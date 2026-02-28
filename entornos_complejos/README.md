# Título del Trabajo
## Información
  - **Alumnos:** Casals, Gloria; Franco, Ángel; 
  - **Asignatura:** Extensiones de Machine Learning
  - **Curso:** 2025/2026
  - **Grupo:** CasalsFranco

## Descripción
  [Breve descripción del trabajo y sus objetivos]

## Estructura

El proyecto sigue una arquitectura modular orientada a objetos para separar claramente políticas, algoritmos de aprendizaje y utilidades:

Casals Franco/

    entornos_complejos/
        README.md                  # Estructura del repositorio
        src/                       # Framework de Reinforcement Learning
        
            agents/                # Clases de los agentes
                __init__.py        # Plantilla abstracta (__init__, get_action, update)
                clase_BASE.py      # Implementación del algoritmo MonteCarlo hacia atrás
                dqn.py             # Implementación del algoritmo DQN
                monte_carlo.py     # Implementación del algoritmo MonteCarlo on-policy
                monte_carlo_off.py # Implementación del algoritmo MonteCarlo off-policy
                q_learning.py      # Implementación del algoritmo tabular Q-Learning
                sarsa.py           # Implementación del algoritmo tabular SARSA
                sarsa_semi_grad.py # Implementación del algoritmo aproximado SARSA semi-gradiente
                sarsa_semi_grad2.py # Implementación del algoritmo aproximado SARSA semi-gradiente con modificaciones en el agente
                
            networks/
                q_network.py       # Implementa una red neuronal profunda en PyTorch para aproximar la función de valor $Q(s, a)$ en entornos de aprendizaje por refuerzo
                
            policies/              # Políticas de toma de decisiones
                __init__.py        # Facilita la importación de las políticas que controlan el equilibrio entre exploración y explotación del agente
                epsilon_decay.py   # Decaimiento del parámetro epsilon
                epsilon_greedy.py  # Lógica de exploración/explotación
                
            utils/                 # Herramientas transversales
                __init__.py        # Facilita la importación de las funciones de la carpeta
                plotter.py         # Generación de gráficas 
                replay_buffer.py   # Modificación del buffer en el algoritmo DQN
                seeder.py          # Fijación de semillas para reproducibilidad estricta
                trainer.py         # Bucle de entrenamiento principal
            
        
        
        

## Instalación y Uso
  Abre el archivo EML_practica1_parte2.ipynb directamente en Colab utilizando el botón "Open in Colab".
  El entorno está configurado con semillas estáticas (seeder.py) en PyTorch, NumPy y Gymnasium para asegurar que los resultados visualizados en la memoria gráfica coincidan exactamente con la ejecución del código . 
  Para ejecutar los experimentos, simplemente selecciona "Entorno de ejecución > Ejecutar todas" en Colab.

## Tecnologías Utilizadas

  -Lenguaje: Python 3.10+
  -Entornos de Simulación: Gymnasium (Blackjack-v1 y CliffWalking-v1 para métodos tabulares y MountainCar-v0 para métodos aproximados).
  -Matemáticas y Estructuras de Datos: NumPy (cálculo vectorial y tablas Q).
  -Deep Learning: PyTorch (para las Redes Neuronales Q-Net en la fase de métodos aproximados) .
