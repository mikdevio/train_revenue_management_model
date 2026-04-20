import pytest
from src.models_math import DeterministicOptimizer

def test_deterministic_capacity_constraint():
    # Configuración de prueba
    config = {'beta': 0.01, 'demand_scale': 1000, 'train_capacity': 100}
    opt = DeterministicOptimizer(config)
    
    results = opt.solve()
    
    # Prueba: El resultado debe ser óptimo
    assert results['status'] == 'OPTIMAL'
    # Prueba: La demanda calculada NO debe superar la capacidad de 100
    assert results['expected_demand'] <= 100 + 1e-5
    # Prueba: El precio debe ser un número positivo
    assert results['best_price'] > 0
