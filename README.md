# Revenue Management System: Optimización de Precios para Transporte Ferroviario

Este sistema implementa un modelo matemático de **Revenue Management Estocástico** para una operadora de trenes. El núcleo del proyecto combina predicciones de demanda mediante Redes Neuronales (RNA) con un motor de optimización en **Gurobi** para determinar la estrategia tarifaria óptima considerando el comportamiento de la competencia.

## 🧠 Descripción del Modelo

El sistema aborda el problema de fijación de precios dinámicos mediante un enfoque de dos etapas:
1.  **Módulo Predictivo (RNA):** Estima las curvas de demanda y la probabilidad de elección del usuario frente a las tarifas de la competencia.
2.  **Módulo de Optimización (Gurobi):** Un modelo de programación matemática estocástica que maximiza el ingreso esperado (Revenue) bajo restricciones de capacidad, horizontes temporales y reglas de negocio.

## 📂 Estructura del Repositorio

```text
├── config/             # Parámetros de Gurobi (MIPGap, TimeLimit) y escenarios estocásticos
├── data/               # Datos históricos de ventas, rutas y precios de la competencia
├── models/             # Modelos de RNA entrenados para predicción de demanda
├── notebooks/          # Experimentación y análisis
│   ├── 01_analisis_competencia.ipynb
│   ├── 02_entrenamiento_rna.ipynb
│   └── 03_validacion_modelo_estocastico.ipynb
├── src/                
│   ├── demand_model.py # Inferencia con la RNA
│   ├── stochastic_opt.py # Formulación matemática en Gurobi
│   ├── engine.py       # Orquestador (Inyecta predicciones en el optimizador)
│   └── simulator.py    # Simulación de Monte Carlo para validación de resultados
├── .env                # Configuración de licencias (Gurobi) y rutas de sistema
└── setup.py            # Instalación del proyecto como módulo local
