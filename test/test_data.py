from src.utils_math import preprocess_features

def test_preprocess_output_shape():
    route = "MAD-BAR"
    date = "2024-12-01"
    features = preprocess_features(route, date)
    
    # Verifica que existan las columnas necesarias para la RNA
    assert 'competitor_prices' in features
    assert isinstance(features['competitor_prices'], list)
