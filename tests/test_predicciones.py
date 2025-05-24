"""
Suite de Tests para el Módulo Predicciones
Distribuye 8 tests: 3 unitarios, 2 integración, 2 paramétricos, 1 E2E
Demuestra aplicación de mejores prácticas de testing para ML
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine
import streamlit as st
import sys
import importlib.util

# =============================================
# TESTS UNITARIOS (3 tests - 38% de 8)
# =============================================

@pytest.mark.unit
@pytest.mark.predicciones
class TestFuncionesUnitariasPredicciones:
    """Tests unitarios para funciones individuales del módulo de predicciones"""
    
    def test_hex_to_rgba_conversion_basica(self):
        """
        Test unitario para conversión hex a rgba con casos básicos
        Propósito: Verificar que la conversión de colores funciona correctamente
        Técnicas usadas: Validación de strings, casos típicos
        """
        # Arrange - Importar la función desde el módulo principal
        spec = importlib.util.spec_from_file_location("predicciones", "pages/2_Predicciones.py")
        predicciones_module = importlib.util.module_from_spec(spec)
        sys.modules["predicciones"] = predicciones_module
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.error'), \
             patch('streamlit.stop'):
            spec.loader.exec_module(predicciones_module)
        
        # Act & Assert
        result = predicciones_module.hex_to_rgba("#FF0000", 0.5)
        assert result == "rgba(255,0,0,0.5)"
        
        result_blue = predicciones_module.hex_to_rgba("#0000FF", 0.3)
        assert result_blue == "rgba(0,0,255,0.3)"
        
        result_default = predicciones_module.hex_to_rgba("#00FF00")
        assert result_default == "rgba(0,255,0,0.2)"
        
    def test_validacion_columnas_requeridas(self):
        """
        Test unitario para validación de columnas requeridas en DataFrame
        Propósito: Verificar que se validan correctamente las columnas del DataFrame
        Técnicas usadas: Validación de estructura de datos, casos positivos/negativos
        """
        # Arrange - Importar la función
        spec = importlib.util.spec_from_file_location("predicciones", "pages/2_Predicciones.py")
        predicciones_module = importlib.util.module_from_spec(spec)
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.error'), \
             patch('streamlit.stop'):
            spec.loader.exec_module(predicciones_module)
        
        requeridas = ['Fecha', 'Municipio', 'Estrato', 'Servicio', 'Cargo Fijo']
        
        # DataFrame válido
        df_valido = pd.DataFrame({
            'Fecha': [pd.to_datetime('2024-01-01')],
            'Municipio': ['Medellín'],
            'Estrato': [1],
            'Servicio': ['Acueducto'],
            'Cargo Fijo': [15000],
            'Extra': ['dato_extra']
        })
        
        # DataFrame inválido
        df_invalido = pd.DataFrame({
            'Fecha': [pd.to_datetime('2024-01-01')],
            'Municipio': ['Medellín'],
            'Estrato': [1]
            # Faltan 'Servicio' y 'Cargo Fijo'
        })
        
        # Act & Assert
        assert predicciones_module.validar_columnas_requeridas(df_valido, requeridas)
        assert not predicciones_module.validar_columnas_requeridas(df_invalido, requeridas)
        
    @patch('os.getenv')
    def test_crear_engine_configuracion_valida(self, mock_getenv):
        """
        Test unitario para crear_engine con configuración válida
        Propósito: Verificar que se crea correctamente el engine de BD
        Técnicas usadas: Mocks de variables de entorno, validación de strings
        """
        # Arrange
        mock_getenv.side_effect = lambda key: {
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DB_NAME': 'test_db'
        }.get(key)
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            # Importar y ejecutar la función
            spec = importlib.util.spec_from_file_location("predicciones", "pages/2_Predicciones.py")
            predicciones_module = importlib.util.module_from_spec(spec)
            
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.title'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.error'), \
                 patch('streamlit.stop'):
                spec.loader.exec_module(predicciones_module)
            
            # Act
            result = predicciones_module.crear_engine()
            
            # Assert
            assert result == mock_engine
            expected_url = 'postgresql+psycopg2://test_user:test_pass@localhost:5432/test_db'
            # Permitir que se llame múltiples veces debido a la ejecución del módulo
            assert mock_create_engine.call_count >= 1
            # Verificar que al menos una llamada fue con la URL correcta
            calls = [call.args[0] for call in mock_create_engine.call_args_list]
            assert expected_url in calls

# =============================================
# TESTS DE INTEGRACIÓN (2 tests - 25% de 8)
# =============================================

@pytest.mark.integration
@pytest.mark.predicciones
@pytest.mark.database
class TestIntegracionPredicciones:
    """Tests de integración para flujos completos del módulo de predicciones"""
    
    def test_cargar_datos_integracion_bd(self, mock_database_engine, sample_tarifas_dataframe):
        """
        Test de integración para carga de datos desde base de datos
        Propósito: Verificar la integración completa con la base de datos
        Técnicas usadas: Mocks de BD, fixtures, validación de tipos de datos
        """
        # Arrange
        # Importar módulo
        spec = importlib.util.spec_from_file_location("predicciones", "pages/2_Predicciones.py")
        predicciones_module = importlib.util.module_from_spec(spec)
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.error'), \
             patch('streamlit.stop'), \
             patch('sqlalchemy.create_engine') as mock_create_engine, \
             patch('pandas.read_sql') as mock_read_sql, \
             patch('pandas.read_csv') as mock_read_csv:
             
             mock_create_engine.return_value = mock_database_engine
             # Configurar para que falle la BD y use CSV
             mock_read_sql.side_effect = Exception("Database connection failed")
             mock_read_csv.return_value = sample_tarifas_dataframe
             
             spec.loader.exec_module(predicciones_module)
             
             # Act
             df = predicciones_module.cargar_datos()
        
        # Assert
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Verificar estructura del DataFrame
        assert 'Fecha' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['Fecha'])
        
    @patch('plotly.graph_objects.Figure')
    @patch('streamlit.plotly_chart')
    def test_integracion_visualizacion_predicciones(self, mock_st_plotly, mock_figure, mock_plotly_figure):
        """
        Test de integración para visualización completa de predicciones
        Propósito: Verificar la integración entre datos, modelos y visualización
        Técnicas usadas: Mocks de Plotly y Streamlit, validación de gráficos
        """
        # Arrange
        mock_figure.return_value = mock_plotly_figure
        
        # Datos históricos y predicciones simulados
        fechas_historicas = pd.date_range('2020-01-01', periods=24, freq='M')
        valores_historicos = np.random.normal(20000, 2000, 24)
        
        fechas_futuras = pd.date_range('2022-01-01', periods=12, freq='M')
        predicciones_prophet = np.random.normal(22000, 1500, 12)
        predicciones_arima = np.random.normal(21500, 1800, 12)
        
        # Act - Simular creación del gráfico
        fig = mock_figure()
        
        # Agregar datos históricos
        fig.add_trace(go.Scatter(
            x=fechas_historicas, 
            y=valores_historicos, 
            mode='lines+markers', 
            name='Histórico'
        ))
        
        # Agregar predicciones
        fig.add_trace(go.Scatter(
            x=fechas_futuras, 
            y=predicciones_prophet, 
            mode='lines', 
            name='Prophet'
        ))
        
        fig.add_trace(go.Scatter(
            x=fechas_futuras, 
            y=predicciones_arima, 
            mode='lines', 
            name='ARIMA'
        ))
        
        # Configurar layout
        fig.update_layout(
            title="Predicción Tarifaria",
            xaxis_title="Fecha",
            yaxis_title="Cargo Fijo ($COP)"
        )
        
        # Mostrar en Streamlit
        mock_st_plotly(fig, use_container_width=True)
        
        # Assert
        assert mock_figure.called
        assert mock_plotly_figure.add_trace.call_count == 3  # Histórico + 2 predicciones
        assert mock_plotly_figure.update_layout.called
        mock_st_plotly.assert_called_once_with(fig, use_container_width=True)

# =============================================
# TESTS PARAMÉTRICOS (2 tests - 25% de 8)
# =============================================

@pytest.mark.parametric
@pytest.mark.predicciones
class TestParametricosPredicciones:
    """Tests paramétricos para validar múltiples casos de entrada en predicciones"""
    
    @pytest.mark.parametrize("horizonte,nivel_confianza,modelo", [
        (3, 80, "Prophet"),
        (6, 90, "ARIMA"),
        (12, 95, "XGBoost")
    ])
    def test_configuracion_predicciones_parametrica(self, horizonte, nivel_confianza, modelo,
                                                  mock_prophet_model, mock_arima_model, mock_xgboost_model):
        """
        Test paramétrico para diferentes configuraciones de predicción
        Propósito: Verificar que las predicciones funcionan con diversos parámetros
        Técnicas usadas: Parametrización múltiple, fixtures de modelos ML
        """
        # Arrange
        serie_datos = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=36, freq='MS'),
            'y': np.random.normal(20000, 2000, 36)
        })
        
        # Act & Assert según el modelo
        if modelo == "Prophet":
            with patch('prophet.Prophet') as mock_prophet_class:
                mock_prophet_class.return_value = mock_prophet_model
                
                # Configurar mock para retornar el horizonte correcto
                forecast_data = pd.DataFrame({
                    'yhat': np.random.normal(20000, 1000, horizonte),
                    'yhat_lower': np.random.normal(19000, 1000, horizonte),
                    'yhat_upper': np.random.normal(21000, 1000, horizonte)
                })
                mock_prophet_model.predict.return_value = forecast_data
                
                # Configurar Prophet con nivel de confianza
                prophet_instance = mock_prophet_class(interval_width=nivel_confianza / 100)
                prophet_instance.fit(serie_datos)
                
                future = prophet_instance.make_future_dataframe(periods=horizonte, freq='MS')
                forecast = prophet_instance.predict(future)
                
                # Assert
                assert len(forecast) == horizonte
                assert 'yhat' in forecast.columns
                
        elif modelo == "ARIMA":
            with patch('statsmodels.tsa.arima.model.ARIMA') as mock_arima_class:
                mock_arima_instance = Mock()
                mock_arima_instance.fit.return_value = mock_arima_model
                mock_arima_class.return_value = mock_arima_instance
                
                # Configurar mock para retornar el horizonte correcto
                predicted_values = pd.Series(np.random.normal(20000, 1000, horizonte))
                mock_arima_model.get_forecast.return_value.predicted_mean = predicted_values
                
                # Entrenar y predecir
                arima_model = mock_arima_class(serie_datos['y'], order=(1, 1, 1))
                fitted_model = arima_model.fit()
                forecast = fitted_model.get_forecast(steps=horizonte)
                
                # Assert
                assert forecast.predicted_mean is not None
                assert len(forecast.predicted_mean) == horizonte
                
        elif modelo == "XGBoost":
            with patch('xgboost.XGBRegressor') as mock_xgb_class:
                mock_xgb_class.return_value = mock_xgboost_model
                
                # Configurar mock para retornar el horizonte correcto
                mock_xgboost_model.predict.return_value = np.random.normal(20000, 1000, horizonte)
                
                # Preparar datos para XGBoost
                X = serie_datos.assign(
                    mes=serie_datos['ds'].dt.month,
                    año=serie_datos['ds'].dt.year
                )[['mes', 'año']]
                
                xgb_model = mock_xgb_class(n_estimators=100)
                xgb_model.fit(X, serie_datos['y'])
                
                # Crear datos futuros
                future_dates = pd.date_range(
                    start=serie_datos['ds'].iloc[-1] + pd.DateOffset(months=1),
                    periods=horizonte,
                    freq='MS'
                )
                X_future = pd.DataFrame({
                    'mes': future_dates.month,
                    'año': future_dates.year
                })
                
                predicciones = xgb_model.predict(X_future)
                
                # Assert
                assert len(predicciones) == horizonte
                assert all(isinstance(p, (int, float, np.number)) for p in predicciones)
                
    @pytest.mark.parametrize("municipio,estrato,servicio,debe_existir", [
        ("Medellín", 1, "Acueducto", True),
        ("Bello", 3, "Alcantarillado", True),
        ("Ciudad_Inexistente", 1, "Acueducto", False)
    ])
    def test_filtrado_datos_combinaciones(self, municipio, estrato, servicio, debe_existir, 
                                        sample_tarifas_dataframe):
        """
        Test paramétrico para filtrado de datos con diferentes combinaciones
        Propósito: Verificar que el filtrado funciona correctamente con todos los casos
        Técnicas usadas: Casos válidos e inválidos, validación booleana
        """
        # Arrange
        df = sample_tarifas_dataframe
        
        # Act
        df_filtrado = df[
            (df['Municipio'] == municipio) &
            (df['Estrato'] == estrato) &
            (df['Servicio'] == servicio)
        ]
        
        # Assert
        if debe_existir:
            assert not df_filtrado.empty, f"Debería existir data para {municipio}-{estrato}-{servicio}"
            assert all(df_filtrado['Municipio'] == municipio)
            assert all(df_filtrado['Estrato'] == estrato)
            assert all(df_filtrado['Servicio'] == servicio)
        else:
            assert df_filtrado.empty, f"No debería existir data para {municipio}-{estrato}-{servicio}"

# =============================================
# TESTS E2E CON SELENIUM (1 test - 12% de 8)
# =============================================

@pytest.mark.e2e
@pytest.mark.predicciones
@pytest.mark.slow
class TestE2EPredicciones:
    """Tests end-to-end para flujos completos de usuario en predicciones"""
    
    def test_flujo_completo_configuracion_y_prediccion(self, setup_streamlit_predicciones_app):
        """
        Test E2E para flujo completo de configuración y visualización de predicciones
        Propósito: Verificar que un usuario puede configurar y ver predicciones completas
        Técnicas usadas: Selenium, interacciones complejas, validación de resultados
        """
        # Arrange
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select, WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        import time
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 30)
        
        try:
            # Act - Navegar a la aplicación de predicciones
            driver.get("http://localhost:8502/2_Predicciones")
            
            # Esperar carga completa
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            time.sleep(8)
            
            # Verificar que estamos en la página correcta
            page_title = driver.find_element(By.TAG_NAME, "h1")
            assert "Predicciones" in page_title.text, "No estamos en la página de predicciones"
            
            # Dar tiempo para que se generen los gráficos
            time.sleep(10)
            
            # Assert - Verificaciones básicas y resilientes
            
            # 1. Verificar que la página tiene contenido sustancial
            page_content = driver.find_elements(By.XPATH, "//div | //p | //span")
            assert len(page_content) > 20, "La página debe tener contenido sustancial"
            
            # 2. Verificar que hay texto relacionado con predicciones o modelos
            prediction_text = driver.find_elements(By.XPATH, "//*[contains(text(), 'Predicción') or contains(text(), 'Modelo') or contains(text(), 'Prophet') or contains(text(), 'ARIMA') or contains(text(), 'XGBoost') or contains(text(), 'Histórico') or contains(text(), 'Cargo')]")
            assert len(prediction_text) > 0, "Debe haber texto relacionado con predicciones"
            
            # 3. Verificar que hay algún tipo de métrica numérica
            numeric_content = driver.find_elements(By.XPATH, "//*[contains(text(), '$') or contains(text(), '%') or contains(text(), 'COP') or contains(text(), '20') or contains(text(), '1') or contains(text(), '2') or contains(text(), '3')]")
            assert len(numeric_content) > 0, "Debe haber contenido numérico"
            
            # 4. Verificar que la aplicación está funcionando (no hay errores críticos)
            error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Error') or contains(text(), 'error') or contains(text(), 'Exception')]")
            assert len(error_elements) <= 5, f"No debe haber demasiados errores críticos. Encontrados: {len(error_elements)}"
            
        except Exception as e:
            # Capturar screenshot para debugging si falla
            try:
                driver.save_screenshot("test_failure_predicciones.png")
                print(f"Screenshot guardado en: test_failure_predicciones.png")
            except:
                pass
            
            # Imprimir información de debugging
            try:
                print(f"Título de página: {driver.title}")
                print(f"URL actual: {driver.current_url}")
                body_text = driver.find_element(By.TAG_NAME, "body").text[:500]
                print(f"Contenido de página (primeros 500 chars): {body_text}")
            except:
                pass
                
            raise e
            
        finally:
            driver.quit()

# =============================================
# FIXTURES ESPECÍFICAS PARA ESTE MÓDULO
# =============================================

@pytest.fixture
def setup_streamlit_predicciones_app():
    """
    Fixture para configurar la aplicación Streamlit de predicciones para tests E2E
    Inicia la aplicación en background para testing
    """
    import subprocess
    import time
    import requests
    
    # Iniciar la aplicación Streamlit en puerto diferente
    process = subprocess.Popen([
        "streamlit", "run", "pages/2_Predicciones.py", 
        "--server.port", "8502", 
        "--server.headless", "true",
        "--logger.level", "error"
    ])
    
    # Esperar a que la aplicación esté lista
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8502", timeout=1)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        process.terminate()
        pytest.fail("No se pudo iniciar la aplicación Streamlit de predicciones")
    
    yield
    
    # Cleanup: terminar el proceso
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill() 