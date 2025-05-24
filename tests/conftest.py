"""
Configuración principal de pytest para el Sistema de Predicción de Tarifas
Incluye fixtures, configuraciones y utilidades compartidas para toda la suite de tests
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point, Polygon
import streamlit as st
import folium
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import sqlite3

# ========== CONFIGURACIÓN GLOBAL ==========

@pytest.fixture(scope="session")
def setup_test_environment():
    """Configuración inicial del entorno de testing"""
    # Configurar variables de entorno para testing
    os.environ['TESTING'] = 'true'
    os.environ['DB_USER'] = 'test_user'
    os.environ['DB_PASSWORD'] = 'test_pass'
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '5432'
    os.environ['DB_NAME'] = 'test_db'
    
    yield
    
    # Cleanup después de todos los tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

@pytest.fixture
def temp_directory():
    """Crea un directorio temporal para tests que requieren archivos"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# ========== FIXTURES PARA DATOS DE PRUEBA ==========

@pytest.fixture
def sample_tarifas_dataframe():
    """Crea un DataFrame de tarifas de prueba con estructura realista"""
    np.random.seed(42)  # Para resultados reproducibles
    
    municipios = ['Medellín', 'Bello', 'Itagüí', 'Envigado', 'Sabaneta']
    estratos = [1, 2, 3, 4, 5, 6]
    servicios = ['Acueducto', 'Alcantarillado']
    años = [2020, 2021, 2022, 2023, 2024]
    
    data = []
    for año in años:
        for municipio in municipios:
            for estrato in estratos:
                for servicio in servicios:
                    # Simular datos realistas con variaciones por municipio y estrato
                    base_tarifa = 15000 + (estrato * 2000) + np.random.normal(0, 1000)
                    
                    data.append({
                        'Año': año,
                        'Fecha': pd.to_datetime(f'{año}-{np.random.randint(1, 13):02d}-01'),
                        'Municipio': municipio,
                        'Estrato': estrato,
                        'Servicio': servicio,
                        'Cargo Fijo': max(0, base_tarifa),
                        'diferencial_por_estrato': np.random.uniform(0.8, 1.5),
                        'indice_carga': np.random.uniform(0.5, 2.0),
                        'ratio_penalizacion': np.random.uniform(0.1, 0.8),
                        'factor_operativo': np.random.uniform(0.9, 1.3),
                        'indice_penalizacion': np.random.uniform(0.2, 1.0),
                        'dispersion_municipal': np.random.uniform(100, 1000),
                        'ratio_municipal': np.random.uniform(0.7, 1.4),
                        'indice_variabilidad': np.random.uniform(0.1, 0.6)
                    })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_geodataframe():
    """Crea un GeoDataFrame de municipios de prueba"""
    municipios = ['Medellín', 'Bello', 'Itagüí', 'Envigado', 'Sabaneta']
    
    # Coordenadas aproximadas del Valle de Aburrá
    coords = [
        [6.25184, -75.56359],  # Medellín
        [6.33732, -75.55290],  # Bello  
        [6.17591, -75.59893],  # Itagüí
        [6.16667, -75.58333],  # Envigado
        [6.15153, -75.61657]   # Sabaneta
    ]
    
    geometries = []
    for lat, lon in coords:
        # Crear polígonos pequeños alrededor de cada punto
        polygon = Polygon([
            (lon - 0.01, lat - 0.01),
            (lon + 0.01, lat - 0.01),
            (lon + 0.01, lat + 0.01),
            (lon - 0.01, lat + 0.01)
        ])
        geometries.append(polygon)
    
    gdf = gpd.GeoDataFrame({
        'MpNombre': municipios,
        'MpNombre_norm': [m.lower().replace(' ', '') for m in municipios],
        'geometry': geometries
    }, crs="EPSG:4326")
    
    return gdf

@pytest.fixture
def mock_database_engine():
    """Mock de engine de base de datos para testing"""
    engine = Mock()
    
    # Configurar el mock para devolver datos de prueba
    def mock_read_sql(query, con):
        if "tarifas_acueductos" in query:
            return sample_tarifas_dataframe()
        return pd.DataFrame()
    
    with patch('pandas.read_sql', side_effect=mock_read_sql):
        yield engine

@pytest.fixture
def mock_streamlit_session_state():
    """Mock del session state de Streamlit"""
    mock_state = {
        'last_indicador': None,
        'map_container': None
    }
    
    with patch.object(st, 'session_state', mock_state):
        yield mock_state

# ========== FIXTURES PARA ARCHIVOS DE PRUEBA ==========

@pytest.fixture
def temp_csv_file(temp_directory, sample_tarifas_dataframe):
    """Crea un archivo CSV temporal con datos de prueba"""
    csv_path = os.path.join(temp_directory, 'test_tarifas.csv')
    sample_tarifas_dataframe.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def temp_shapefile(temp_directory, sample_geodataframe):
    """Crea un shapefile temporal con datos de prueba"""
    shp_dir = os.path.join(temp_directory, 'shp')
    os.makedirs(shp_dir)
    shp_path = os.path.join(shp_dir, 'municipios.shp')
    sample_geodataframe.to_file(shp_path)
    return shp_path

@pytest.fixture
def mock_file_paths(temp_csv_file, temp_shapefile):
    """Mock de rutas de archivos para testing"""
    paths = {
        'csv_path': temp_csv_file,
        'shp_path': os.path.dirname(temp_shapefile)
    }
    
    with patch('os.path.exists', return_value=True):
        yield paths

# ========== FIXTURES PARA MODELOS DE ML ==========

@pytest.fixture
def mock_prophet_model():
    """Mock del modelo Prophet para testing"""
    mock_prophet = Mock()
    
    # Configurar predicciones de prueba
    mock_forecast = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=12, freq='M'),
        'yhat': np.random.normal(20000, 1000, 12),
        'yhat_lower': np.random.normal(18000, 1000, 12),
        'yhat_upper': np.random.normal(22000, 1000, 12)
    })
    
    mock_prophet.predict.return_value = mock_forecast
    mock_prophet.make_future_dataframe.return_value = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=60, freq='M')
    })
    
    return mock_prophet

@pytest.fixture
def mock_arima_model():
    """Mock del modelo ARIMA para testing"""
    mock_arima = Mock()
    
    # Mock del forecast
    mock_forecast_result = Mock()
    mock_forecast_result.predicted_mean = pd.Series(np.random.normal(20000, 1000, 12))
    mock_forecast_result.conf_int.return_value = pd.DataFrame({
        'lower': np.random.normal(18000, 1000, 12),
        'upper': np.random.normal(22000, 1000, 12)
    })
    
    mock_arima.get_forecast.return_value = mock_forecast_result
    
    return mock_arima

@pytest.fixture
def mock_xgboost_model():
    """Mock del modelo XGBoost para testing"""
    mock_xgb = Mock()
    mock_xgb.predict.return_value = np.random.normal(20000, 1000, 12)
    return mock_xgb

# ========== FIXTURES PARA VISUALIZACIONES ==========

@pytest.fixture
def mock_folium_map():
    """Mock de mapa Folium para testing"""
    mock_map = Mock(spec=folium.Map)
    
    # Configurar métodos del mapa
    mock_map.add_child = Mock()
    mock_map.save = Mock()
    
    with patch('folium.Map', return_value=mock_map):
        yield mock_map

@pytest.fixture
def mock_plotly_figure():
    """Mock de figura Plotly para testing"""
    mock_fig = Mock()
    mock_fig.add_trace = Mock()
    mock_fig.update_layout = Mock()
    mock_fig.update_xaxes = Mock()
    
    return mock_fig

# ========== DATOS PARAMÉTRICOS PARA TESTING ==========

@pytest.fixture(params=[
    'diferencial_por_estrato', 'indice_carga', 'ratio_penalizacion',
    'factor_operativo', 'indice_penalizacion', 'dispersion_municipal',
    'ratio_municipal', 'indice_variabilidad'
])
def indicador_parametrico(request):
    """Parámetros para testing de todos los indicadores"""
    return request.param

@pytest.fixture(params=[
    ('Medellín', 1, 'Acueducto'),
    ('Bello', 3, 'Alcantarillado'),
    ('Itagüí', 5, 'Acueducto'),
    ('Envigado', 2, 'Alcantarillado'),
    ('Sabaneta', 4, 'Acueducto')
])
def filtros_parametricos(request):
    """Parámetros para testing de diferentes combinaciones de filtros"""
    return request.param

@pytest.fixture(params=[3, 6, 12, 18, 24])
def horizontes_prediccion(request):
    """Parámetros para testing de diferentes horizontes de predicción"""
    return request.param

@pytest.fixture(params=[80, 90, 95, 99])
def niveles_confianza(request):
    """Parámetros para testing de diferentes niveles de confianza"""
    return request.param

# ========== UTILIDADES PARA TESTING ==========

@pytest.fixture
def assert_dataframe_structure():
    """Utilidad para validar estructura de DataFrames"""
    def _assert_structure(df, expected_columns, expected_types=None):
        """
        Valida que un DataFrame tenga la estructura esperada
        
        Args:
            df: DataFrame a validar
            expected_columns: Lista de columnas esperadas
            expected_types: Dict opcional con tipos esperados {columna: tipo}
        """
        assert isinstance(df, pd.DataFrame), "El objeto debe ser un DataFrame"
        assert not df.empty, "El DataFrame no debe estar vacío"
        
        # Verificar columnas
        missing_cols = set(expected_columns) - set(df.columns)
        assert not missing_cols, f"Faltan columnas: {missing_cols}"
        
        # Verificar tipos si se especifican
        if expected_types:
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    assert df[col].dtype == expected_type or pd.api.types.is_dtype_equal(df[col].dtype, expected_type), \
                        f"Columna {col} tiene tipo {df[col].dtype}, se esperaba {expected_type}"
    
    return _assert_structure

@pytest.fixture
def assert_map_properties():
    """Utilidad para validar propiedades de mapas Folium"""
    def _assert_map(map_obj, expected_center=None, expected_zoom=None):
        """
        Valida propiedades básicas de un mapa Folium
        
        Args:
            map_obj: Objeto de mapa Folium
            expected_center: Coordenadas esperadas del centro [lat, lon]
            expected_zoom: Nivel de zoom esperado
        """
        assert map_obj is not None, "El mapa no debe ser None"
        
        if expected_center:
            assert hasattr(map_obj, 'location'), "El mapa debe tener ubicación"
            
        if expected_zoom:
            assert hasattr(map_obj, 'zoom_start'), "El mapa debe tener zoom inicial"
    
    return _assert_map

# ========== MARKERS Y PARAMETRIZACIÓN ==========

# Markers para categorización de tests
pytestmark = pytest.mark.sistema_tarifas

def pytest_configure(config):
    """Configuración de markers personalizados"""
    config.addinivalue_line("markers", "unit: Tests unitarios")
    config.addinivalue_line("markers", "integration: Tests de integración")
    config.addinivalue_line("markers", "parametric: Tests paramétricos")
    config.addinivalue_line("markers", "e2e: Tests end-to-end")
    config.addinivalue_line("markers", "visor_geografico: Tests del visor geográfico")
    config.addinivalue_line("markers", "predicciones: Tests del módulo de predicciones")
    config.addinivalue_line("markers", "slow: Tests que tardan más tiempo")
    config.addinivalue_line("markers", "database: Tests que requieren base de datos") 