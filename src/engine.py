import argparse
import logging
import sys
from typing import Dict, Any

# Importaciones locales (asumiendo que instalaste el proyecto con pip install -e .)
from src.models_rna import DemandPredictor
from src.optimizer import TrainRevenueOptimizer
from src.models_math import DeterministicOptimizer
from src.utils_math import load_config, preprocess_features

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RevenueEngine")

class RevenueEngine:
    """
    Orquestador principal que conecta la predicción de demanda con 
    los diferentes motores de optimización.
    """
    def __init__(self, config_path: str, model_type: str = "stochastic"):
        self.config = load_config(config_path)
        self.model_type = model_type
        
        # 1. Inicializar el predictor (RNA)
        self.predictor = DemandPredictor(model_path=self.config['model_path'])
        
        # 2. Selección de la estrategia de optimización (Patrón Strategy)
        if model_type == "stochastic":
            self.optimizer = TrainRevenueOptimizer(self.config)
            logger.info("Usando motor de optimización ESTOCÁSTICO (Gurobi).")
        elif model_type == "deterministic":
            self.optimizer = DeterministicOptimizer(self.config)
            logger.info("Usando motor de optimización DETERMINISTA (SciPy).")
        else:
            raise ValueError(f"Modelo {model_type} no reconocido.")

    def run(self, route_id: str, departure_date: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo: Datos -> RNA -> Optimización -> Resultados.
        """
        try:
            logger.info(f"Iniciando proceso para ruta {route_id} en fecha {departure_date}...")

            # Paso 1: Preprocesamiento de datos (Competencia, clima, etc.)
            features = preprocess_features(route_id, departure_date)

            # Paso 2: Inferencia de la RNA (Predictor)
            # El predictor devuelve parámetros de demanda (media, varianza o escala)
            demand_params = self.predictor.predict_scenario(features)
            logger.info(f"Predicción de demanda completada.")

            # Paso 3: Optimización Matemática
            # Ambos optimizadores deben seguir la misma interfaz con el método .solve()
            results = self.optimizer.solve(
                demand_params=demand_params,
                capacity=self.config['train_capacity'],
                competitor_prices=features.get('competitor_prices', [])
            )

            if results['status'] == 'OPTIMAL':
                logger.info(f"Éxito. Precio Sugerido: {results['best_price']:.2f}")
                return results
            else:
                logger.warning("El optimizador no pudo encontrar una solución óptima.")
                return results

        except Exception as e:
            logger.error(f"Error crítico en el pipeline: {str(e)}")
            raise

def main():
    """
    Punto de entrada de CLI (Interfaz de Línea de Comandos)
    """
    parser = argparse.ArgumentParser(description="Revenue Management Engine for Trains")
    
    # Argumentos obligatorios
    parser.add_argument("--route", type=str, required=True, help="ID de la ruta (ej. MAD-BAR)")
    parser.add_argument("--date", type=str, required=True, help="Fecha de salida (YYYY-MM-DD)")
    
    # Argumentos opcionales
    parser.add_argument(
        "--model", 
        type=str, 
        default="stochastic", 
        choices=["stochastic", "deterministic"],
        help="Tipo de modelo matemático a ejecutar"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/settings.yaml", 
        help="Ruta al archivo de configuración"
    )

    args = parser.parse_args()

    # Ejecución
    engine = RevenueEngine(config_path=args.config, model_type=args.model)
    engine.run(route_id=args.route, departure_date=args.date)

if __name__ == "__main__":
    main()
