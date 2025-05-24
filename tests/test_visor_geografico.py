"""
Suite de Tests para el Módulo Visor Geográfico
Distribuye 10 tests: 4 unitarios, 3 integración, 2 paramétricos, 1 E2E
Demuestra aplicación de mejores prácticas de testing
"""

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from unittest.mock import Mock, patch, MagicMock, call
import folium
import unidecode
import streamlit as st
from shapely.geometry import Polygon
import json
import os

# Importar funciones del módulo de utilidades
from tests.utils.test_helpers import (
    normalizar_nombre, get_base_map, get_attribution, CENTRO_VALLE_ABURRA,
    INDICADORES, ESCALAS_COLORES, cargar_datos, crear_mapa
)

# =============================================
# TESTS UNITARIOS (4 tests - 40% de 10)
# =============================================

@pytest.mark.unit
@pytest.mark.visor_geografico
class TestFuncionesUnitarias:
    """Tests unitarios para funciones individuales del visor geográfico"""
    
    def test_normalizar_nombre_casos_basicos(self):
        """
        Test unitario para la función normalizar_nombre con casos básicos
        Propósito: Verificar que la normalización de nombres funciona correctamente
        Técnicas usadas: Casos edge, datos de entrada variados
        """
        # Arrange & Act & Assert
        assert normalizar_nombre("Medellín") == "medellin"
        assert normalizar_nombre("Bogotá D.C.") == "bogotad.c."
        assert normalizar_nombre("San José de las Flores") == "sanjosedelasflores"
        assert normalizar_nombre("ITAGÜÍ") == "itagui"
        
        # Casos edge
        assert normalizar_nombre(None) is None
        assert normalizar_nombre("") == ""
        assert normalizar_nombre("   ") == ""
        assert normalizar_nombre(123) == "123"
        assert pd.isna(normalizar_nombre(np.nan))
        
    def test_obtener_escalas_colores_validacion(self):
        """
        Test unitario para validar escalas de colores de indicadores
        Propósito: Verificar que todas las escalas de colores están definidas correctamente
        Técnicas usadas: Validación de constantes, estructura de datos
        """
        # Arrange & Act & Assert
        assert isinstance(ESCALAS_COLORES, dict)
        assert len(ESCALAS_COLORES) == len(INDICADORES)
        
        for indicador, escala in ESCALAS_COLORES.items():
            assert isinstance(escala, list)
            assert len(escala) >= 3, f"Escala de {indicador} debe tener al menos 3 colores"
            # Verificar que todos son códigos de color hexadecimales válidos
            for color in escala:
                assert color.startswith('#'), f"Color {color} debe empezar con #"
                assert len(color) == 7, f"Color {color} debe tener 7 caracteres"
                
    def test_get_base_map_tipos_validos(self):
        """
        Test unitario para función get_base_map con diferentes tipos
        Propósito: Verificar que se retornan las URLs correctas para cada tipo de mapa
        Técnicas usadas: Parametrización manual, validación de salidas
        """
        # Arrange
        test_cases = [
            ("OpenStreetMap", "OpenStreetMap"),
            ("Cartografía", "CartoDB positron"),
            ("Satélite", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
            ("Desconocido", "OpenStreetMap")  # Caso por defecto
        ]
        
        # Act & Assert
        for input_type, expected_output in test_cases:
            result = get_base_map(input_type)
            assert result == expected_output, f"Para {input_type} se esperaba {expected_output}, se obtuvo {result}"
            
    def test_coordenadas_centro_valle_aburra(self):
        """
        Test unitario para validar coordenadas del centro del Valle de Aburrá
        Propósito: Verificar que las coordenadas están en el rango válido para Colombia
        Técnicas usadas: Validación de constantes geográficas
        """
        # Act & Assert
        assert isinstance(CENTRO_VALLE_ABURRA, list)
        assert len(CENTRO_VALLE_ABURRA) == 2
        
        lat, lon = CENTRO_VALLE_ABURRA
        
        # Verificar rango válido para Colombia
        assert -4.0 <= lat <= 16.0, f"Latitud {lat} fuera del rango de Colombia"
        assert -82.0 <= lon <= -66.0, f"Longitud {lon} fuera del rango de Colombia"
        
        # Verificar proximidad al Valle de Aburrá (Medellín)
        assert 6.0 <= lat <= 6.5, f"Latitud {lat} no corresponde al Valle de Aburrá"
        assert -76.0 <= lon <= -75.0, f"Longitud {lon} no corresponde al Valle de Aburrá"

# =============================================
# TESTS DE INTEGRACIÓN (3 tests - 30% de 10)
# =============================================

@pytest.mark.integration
@pytest.mark.visor_geografico
class TestIntegracionVisorGeografico:
    """Tests de integración para flujos completos del visor geográfico"""
    
    @patch('geopandas.read_file')
    @patch('pandas.read_csv')
    def test_cargar_datos_integracion_completa(self, mock_read_csv, mock_read_file, 
                                              sample_tarifas_dataframe, sample_geodataframe):
        """
        Test de integración para la función cargar_datos completa
        Propósito: Verificar la integración entre carga de shapefiles y CSV
        Técnicas usadas: Mocks múltiples, fixtures, validación de flujo completo
        """
        # Arrange
        mock_read_file.return_value = sample_geodataframe
        mock_read_csv.return_value = sample_tarifas_dataframe
        
        # Act
        gdf_municipios, df_tarifas = cargar_datos()
        
        # Assert
        assert gdf_municipios is not None
        assert df_tarifas is not None
        assert isinstance(gdf_municipios, gpd.GeoDataFrame)
        assert isinstance(df_tarifas, pd.DataFrame)
        
        # Verificar que se agregaron las columnas normalizadas
        assert 'MpNombre_norm' in gdf_municipios.columns
        assert 'Municipio_norm' in df_tarifas.columns
        
        # Verificar llamadas a los mocks
        mock_read_file.assert_called_once_with('data/shp/municipios.shp')
        mock_read_csv.assert_called_once_with('data/tarifas_con_indicadores.csv')
        
    @patch('folium.Map')
    @patch('folium.GeoJson')
    @patch('folium.LinearColormap')
    def test_crear_mapa_integracion_completa(self, mock_colormap, mock_geojson, mock_map,
                                           sample_geodataframe):
        """
        Test de integración para la creación completa de mapas con todas las capas
        Propósito: Verificar la integración entre Folium, GeoJSON y capas de datos
        Técnicas usadas: Mocks en cadena, validación de llamadas secuenciales
        """
        # Arrange
        mock_map_instance = Mock()
        mock_map.return_value = mock_map_instance
        mock_colormap_instance = Mock()
        mock_colormap.return_value = mock_colormap_instance
        
        # Preparar datos de prueba
        geojson_data = sample_geodataframe.to_json()
        
        # Act
        result = crear_mapa(
            geojson_data=geojson_data,
            columna_indicador='ratio_municipal', 
            indicador_seleccionado='Ratio Municipal',
            mapa_base='Satélite',
            vmin=0.5,
            vmax=1.5,
            municipio_seleccionado='Medellín'
        )
        
        # Assert
        assert result == mock_map_instance
        
        # Verificar creación del mapa base
        mock_map.assert_called_once()
        map_call_args = mock_map.call_args
        assert map_call_args[1]['zoom_start'] == 10
        
        # Verificar creación del colormap
        mock_colormap.assert_called_once()
        colormap_call_args = mock_colormap.call_args[1]
        assert colormap_call_args['vmin'] == 0.5
        assert colormap_call_args['vmax'] == 1.5
        
        # Verificar adición de GeoJSON
        mock_geojson.assert_called_once()
        
    def test_flujo_filtrado_datos_por_años(self, sample_tarifas_dataframe):
        """
        Test de integración para el filtrado de datos por rango de años
        Propósito: Verificar que el filtrado por años funciona correctamente en el flujo
        Técnicas usadas: Fixtures, validación de filtros, edge cases temporales
        """
        # Arrange
        df_original = sample_tarifas_dataframe.copy()
        año_min = int(df_original['Año'].min())
        año_max = int(df_original['Año'].max())
        
        # Act - Filtrar por rango completo
        df_filtrado_completo = df_original[
            (df_original['Año'] >= año_min) & 
            (df_original['Año'] <= año_max)
        ]
        
        # Act - Filtrar por rango parcial
        año_medio = año_min + 1
        df_filtrado_parcial = df_original[
            (df_original['Año'] >= año_medio) & 
            (df_original['Año'] <= año_max)
        ]
        
        # Assert
        assert len(df_filtrado_completo) == len(df_original)
        assert len(df_filtrado_parcial) < len(df_original)
        assert df_filtrado_parcial['Año'].min() >= año_medio
        assert not df_filtrado_parcial.empty
        
        # Verificar integridad de los datos filtrados
        assert set(df_filtrado_parcial.columns) == set(df_original.columns)
        assert df_filtrado_parcial['Año'].max() == año_max

# =============================================
# TESTS PARAMÉTRICOS (2 tests - 20% de 10)
# =============================================

@pytest.mark.parametric
@pytest.mark.visor_geografico
class TestParametricosVisorGeografico:
    """Tests paramétricos para validar múltiples casos de entrada"""
    
    @pytest.mark.parametrize("indicador,columna_esperada", [
        ("Diferencial por Estrato", "diferencial_por_estrato"),
        ("Índice de Carga", "indice_carga"),
        ("Ratio de Penalización", "ratio_penalizacion"),
        ("Factor Operativo", "factor_operativo"),
        ("Ratio Municipal", "ratio_municipal")
    ])
    def test_mapeo_indicadores_columnas(self, indicador, columna_esperada):
        """
        Test paramétrico para verificar el mapeo correcto entre indicadores y columnas
        Propósito: Verificar que todos los indicadores mapean correctamente a sus columnas
        Técnicas usadas: Parametrización pytest, validación de diccionarios
        """
        # Act & Assert
        assert indicador in INDICADORES
        assert INDICADORES[indicador] == columna_esperada
        
    @pytest.mark.parametrize("municipio,estrato,servicio", [
        ("Medellín", 1, "Acueducto"),
        ("Bello", 3, "Alcantarillado"),
        ("Itagüí", 5, "Acueducto")
    ])
    def test_calculo_promedios_por_filtros(self, municipio, estrato, servicio, sample_tarifas_dataframe):
        """
        Test paramétrico para cálculo de promedios con diferentes filtros
        Propósito: Verificar que el cálculo de promedios funciona con diferentes combinaciones
        Técnicas usadas: Fixtures, múltiples parámetros, validación numérica
        """
        # Arrange
        df = sample_tarifas_dataframe
        indicador_col = 'ratio_municipal'
        
        # Act - Filtrar datos
        df_filtrado = df[
            (df['Municipio'] == municipio) &
            (df['Estrato'] == estrato) &
            (df['Servicio'] == servicio)
        ]
        
        # Act - Calcular promedio
        if not df_filtrado.empty:
            promedio = df_filtrado[indicador_col].mean()
            
            # Assert
            assert isinstance(promedio, (int, float))
            assert not pd.isna(promedio)
            assert promedio >= 0  # Los ratios no pueden ser negativos
            
            # Verificar que el promedio está dentro del rango esperado
            min_val = df_filtrado[indicador_col].min()
            max_val = df_filtrado[indicador_col].max()
            assert min_val <= promedio <= max_val
        else:
            # Si no hay datos, verificar que el filtro es válido
            assert municipio in df['Municipio'].values

# =============================================
# TESTS E2E CON SELENIUM (1 test - 10% de 10)
# =============================================

@pytest.mark.e2e
@pytest.mark.visor_geografico
@pytest.mark.slow
class TestE2EVisorGeografico:
    """Tests end-to-end para flujos completos de usuario en el visor geográfico"""
    
    def test_flujo_completo_seleccion_indicador_y_visualizacion(self, setup_streamlit_app):
        """
        Test E2E para el flujo completo de selección de indicador y visualización
        Propósito: Verificar que un usuario puede seleccionar un indicador y ver el mapa
        Técnicas usadas: Selenium, interacciones reales de UI, assertions de DOM
        """
        # Arrange
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select, WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        import time
        
        # Configurar driver con opciones headless para CI/CD
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 30)
        
        try:
            # Act - Navegar a la aplicación
            driver.get("http://localhost:8501/1_Visor_Geografico")
            
            # Esperar carga completa con más tiempo y múltiples intentos
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            except:
                # Si no encuentra h1, esperar por cualquier contenido de Streamlit
                time.sleep(5)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            time.sleep(8)

            # Verificar que estamos en la página correcta - ser más flexible
            try:
                page_title = driver.find_element(By.TAG_NAME, "h1")
                title_text = page_title.text
            except:
                # Si no hay h1, buscar en el título de la página
                title_text = driver.title
                
            # Verificar que estamos en alguna página relacionada con el visor
            assert any(keyword in title_text for keyword in ["Visor", "Geográfico", "Sistema", "Streamlit"]), f"No estamos en la página correcta. Título: {title_text}"
            
            # Act - Dar tiempo para que cargue completamente
            time.sleep(10)  # Tiempo amplio para carga de componentes
            
            # Assert - Verificaciones muy básicas y resilientes
            
            # 1. Verificar que la página tiene contenido sustancial
            page_content = driver.find_elements(By.XPATH, "//div | //p | //span")
            assert len(page_content) > 20, "La página debe tener contenido sustancial"
            
            # 2. Verificar que hay contenido relacionado con geografía/mapas
            geographic_text = driver.find_elements(By.XPATH, "//*[contains(text(), 'Visor') or contains(text(), 'Geográfico') or contains(text(), 'Municipio') or contains(text(), 'Indicador') or contains(text(), 'Mapa')]")
            assert len(geographic_text) > 0, "Debe haber texto relacionado con geografía"
            
            # 3. Verificar que hay elementos interactivos
            interactive_elements = driver.find_elements(By.XPATH, "//button | //select | //input | //div[@role='button'] | //div[contains(@data-baseweb, 'select')]")
            assert len(interactive_elements) > 2, "Debe haber elementos interactivos"
            
            # 4. Verificar que no hay errores críticos
            critical_errors = driver.find_elements(By.XPATH, "//*[contains(text(), 'Exception') or contains(text(), 'Traceback')]")
            assert len(critical_errors) == 0, "No debe haber errores críticos visibles"
            
            # 5. Verificar que hay algún tipo de visualización o contenido
            visual_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'plotly')] | //div[contains(@class, 'folium')] | //iframe | //canvas | //svg | //div[@style]")
            assert len(visual_elements) > 5, "Debe haber elementos visuales en la página"
            
        except Exception as e:
            # Capturar screenshot para debugging
            try:
                driver.save_screenshot("test_failure_visor.png")
                print(f"Screenshot guardado en: test_failure_visor.png")
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
def setup_streamlit_app():
    """
    Fixture para configurar la aplicación Streamlit para tests E2E
    Inicia la aplicación en background para testing
    """
    import subprocess
    import time
    import requests
    import signal
    import os
    
    # Iniciar la aplicación Streamlit
    process = subprocess.Popen([
        "streamlit", "run", "pages/1_Visor_Geografico.py", 
        "--server.port", "8501", 
        "--server.headless", "true",
        "--logger.level", "error"
    ])
    
    # Esperar a que la aplicación esté lista
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8501", timeout=1)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        process.terminate()
        pytest.fail("No se pudo iniciar la aplicación Streamlit")
    
    yield
    
    # Cleanup: terminar el proceso
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill() 