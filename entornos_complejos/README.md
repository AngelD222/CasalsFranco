# Título del Trabajo
## Información
  - **Alumnos:** Apellido1, Nombre1; Apellido2, Nombre2; Apellido3, Nombre3
  - **Asignatura:** Nombre de la asignatura
  - **Curso:** 2025/2026
  - **Grupo:** NombreGrupo (p.e. ap1ap2ap3m, usando los apellidos)

## Descripción
  [Breve descripción del trabajo y sus objetivos]

## Estructura

El proyecto sigue una arquitectura modular orientada a objetos para separar claramente políticas, algoritmos de aprendizaje y utilidades.

NombreGrupo/
│
├── EML_practica1_parte2.ipynb  # Cuaderno Jupyter principal con los experimentos
├── README.md                   # Documentación principal del repositorio
│
└── src/                        # Framework de Reinforcement Learning
    ├── __init__.py
    │
    ├── agents/                 # Clases de los agentes (El "Cerebro")
    │   ├── __init__.py
    │   ├── clase_BASE.py       # Plantilla abstracta (__init__, get_action, update)
    │   └── monte_carlo.py      # Implementación del algoritmo MonteCarlo hacia atrás
    │
    ├── policies/               # Políticas de toma de decisiones
    │   ├── __init__.py
    │   ├── epsilon_greedy.py   # Lógica de exploración/explotación
    │   └── epsilon_decay.py    # Decaimiento del parámetro epsilon
    │
    └── utils/                  # Herramientas transversales
        ├── __init__.py
        ├── seeder.py           # Fijación de semillas para reproducibilidad estricta
        ├── trainer.py          # Bucle de entrenamiento principal
        └── plotter.py          # Generación de gráficas (recompensas y métrica f(t))
        

## Instalación y Uso
  Abre el archivo EML_practica1_parte2.ipynb directamente en Colab utilizando el botón "Open in Colab".
  El entorno está configurado con semillas estáticas (seeder.py) en PyTorch, NumPy y Gymnasium para asegurar que los resultados visualizados en la memoria gráfica coincidan exactamente con la ejecución del código . 
  Para ejecutar los experimentos, simplemente selecciona "Entorno de ejecución > Ejecutar todas" en Colab.

## Tecnologías Utilizadas

  -Lenguaje: Python 3.10+
  -Entornos de Simulación: Gymnasium (FrozenLake-v1 para métodos tabulares y Flappy-bird-gymnasium para métodos aproximados).
  -Matemáticas y Estructuras de Datos: NumPy (cálculo vectorial y tablas Q).
  -Deep Learning: PyTorch (para las Redes Neuronales Q-Net en la fase de métodos aproximados) .
