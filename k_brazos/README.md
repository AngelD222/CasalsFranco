# Práctica 1, Parte 1: Aprendizaje en el bandido de k-brazos

## Información
  - **Alumnos:** Casals, Gloria; Franco, Ángel; 
  - **Asignatura:** Extensiones de Machine Learning
  - **Curso:** 2025/2026
  - **Grupo:** CasalsFranco

## Descripción


Este repositorio contiene la implementación estructurada y el estudio comparativo de diversos algoritmos clásicos de aprendizaje por refuerzo orientados a resolver el problema del **Bandido Multibrazo (k-Armed Bandit)**. 

El desafío central es encontrar el equilibrio perfecto entre la exploración de nuevas alternativas y la explotación. Vamos a contrastar el rendimiento de las políticas mediante simulaciones de Monte Carlo (múltiples ejecuciones independientes), analizando métricas como la evolución de la recompensa media, la tasa de éxito óptimo y el rechazo acumulado. También estudiaremos el Error Cuadrático Medio (MSE) en las estimaciones de valor de los agentes para comprender cómo la escala de las recompensas y la elección de hiperparámetros afectan a su convergencia.


## Estructura

El repositorio sigue un patrón de diseño orientado a objetos, separando el entorno (brazos), los agentes (algoritmos) y las herramientas de evaluación (visualización).

Casals Franco/

    k_brazos/
        README.md                               # Estructura del repositorio
        src/                                    # Framework de Reinforcement Learning
        
            algorithms/                         # Implementación de las estrategias de los agentes
                __init__.py                     # Facilita la importación
                algorithm.py                    # Clase abstracta base para los algoritmos
                epsilon_decay.py                # variante Decay
                epsilon_greedy.py               # Algoritmo Epsilon-Greedy
                softmax.py                      # Algoritmo Softmax
                ucb.py                          # Algoritmo UCB1
                ucb2.py                         # Algoritmo UCB2
                
             arms/                              # Implementación de los entornos y distribuciones
                __init__.py                     # Facilita la importación
                arm.py                          # Clase abstracta base para los brazos
                armbernoulli.py                 # Brazo con distribución de Bernoulli
                armbinomial.py                  # Brazo con distribución Binomial B(n, p)
                armnormal.py                    # Brazo con distribución Normal N(μ, σ)
                bandit.py                       # Clase principal que agrupa k brazos (Entorno)
                
            plotting/                           # Herramientas transversales
                __init__.py                     # Facilita la importación de las funciones para graficas
                plotting.py                     # Funciones para generar gráficas (MSE, Regret, etc...)
            
        main.ipynb                              # Fichero principal. Breve introducción del problema y enlace a los estudios.
        notebook_1_distribucion_normal.ipynb    # Estudio 1: comportamiento de los algoritmos en brazos de distribución Normal
        notebook_2_distribucion_binomial.ipynb  # Estudio 2: comportamiento de los algoritmos en brazos de distribución Binomial
        notebook_3_distribucion_bernoulli.ipynb # Estudio 3: comportamiento de los algoritmos en brazos de distribución Bernoulli
        
        

## Instalación y Uso

Este proyecto está diseñado para ejecutarse principalmente en entornos de Jupyter Notebook o Google Colab, aunque también puede utilizarse de forma local.

* `main.ipynb`: Fichero principal que accede a los distintos *notebooks* (`main.ipynb`) de cada parte del proyecto, y estos *notebooks* a su vez contiene enlaces a todos los *notebooks* que se han utilizado para experimentar en cada parte.


## Tecnologías Utilizadas

* **Lenguaje:** [Python 3.8+]
* **Computación Numérica:** [NumPy]
* **Visualización de Datos:**
  * [Matplotlib]
  * [Seaborn]
* **Entornos de Desarrollo:**
    * [Jupyter Notebook / Google Colab]
    * [Git & GitHub](https://github.com/)
