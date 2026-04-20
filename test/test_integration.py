from src.engine import RevenueEngine

def test_engine_run_deterministic():
    # Usamos un archivo de config de prueba o un diccionario
    engine = RevenueEngine(config_path="config/settings_test.yaml", model_type="deterministic")
    results = engine.run(route_id="TEST", departure_date="2024-01-01")
    
    assert results is not None
    assert 'best_price' in results
