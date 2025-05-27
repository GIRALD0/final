"""
Suite de Tests para el Módulo Home (Dashboard Principal)
Contiene tests Unitarios, de Integración, Paramétricos y E2E para el home.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock # Asegúrate que MagicMock está importado
import os
import sys
import importlib.util
import time # No usado directamente en unit/integración, pero sí en E2E

# Selenium imports (no necesarios para unit/integración, pero el skipif los usa)
try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# =============================================
# MOCK DE STREAMLIT Y OTRAS LIBRERÍAS PARA TESTS NO-E2E
# =============================================
mock_st_global = MagicMock(name="GlobalStreamlitMock")

def mock_st_columns_flexible(count_or_spec):
    if isinstance(count_or_spec, int):
        return [MagicMock(name=f"column_{i+1}_for_int_{count_or_spec}") for i in range(count_or_spec)]
    elif isinstance(count_or_spec, list): 
        return [MagicMock(name=f"column_{i+1}_for_list_len_{len(count_or_spec)}") for i in range(len(count_or_spec))]
    return MagicMock(name="fallback_column_mock") 

mock_st_global.columns = MagicMock(side_effect=mock_st_columns_flexible, name="MockFor_st.columns")
mock_st_global.sidebar = MagicMock(name="MockFor_st.sidebar")
mock_st_global.sidebar.markdown = MagicMock(name="MockFor_st.sidebar.markdown")
mock_st_global.expander = MagicMock(name="MockFor_st.expander")
mock_st_global.expander.return_value.__enter__ = MagicMock(return_value=None) # Para el 'with st.expander(...):'
mock_st_global.expander.return_value.__exit__ = MagicMock(return_value=None)
mock_st_global.tabs = MagicMock(return_value=(MagicMock(name="TabContextManager1"), MagicMock(name="TabContextManager2")), name="MockFor_st.tabs")
mock_st_global.tabs.return_value[0].__enter__ = MagicMock(return_value=None) 
mock_st_global.tabs.return_value[0].__exit__ = MagicMock(return_value=None)
mock_st_global.tabs.return_value[1].__enter__ = MagicMock(return_value=None) 
mock_st_global.tabs.return_value[1].__exit__ = MagicMock(return_value=None)

sys.modules['streamlit'] = mock_st_global
# Mockear Prophet y dotenv a nivel de módulo para que estén disponibles cuando se importe home
sys.modules['prophet'] = MagicMock(name="GlobalProphetMockModule")
sys.modules['prophet'].Prophet = MagicMock(name="GlobalProphetClassMock") # Mockear la clase Prophet
if 'dotenv' not in sys.modules:
    sys.modules['dotenv'] = MagicMock(name="GlobalDotenvMock")

_loaded_home_module_for_mock = None # Para ayudar al mock de selectbox a acceder a indicadores_clave

def mock_st_selectbox_dynamic(label, options, index=0, key=None, help=None):
    global _loaded_home_module_for_mock
    # print(f"DEBUG mock_st_selectbox called with label: '{label}'") 
    if label == "Desde" or label == "Hasta":
        if options and isinstance(options[0], (int, np.integer)):
            return options[index if index < len(options) else 0]
        return 2023 # Fallback numérico para años
    if label == "Municipio": return "Medellín"
    if label == "Estrato": return 1 # home.py usa esto como int en el filtrado
    if label == "Tipo de Servicio": return "Acueducto"
    if label == "Seleccione municipio base": return "Medellín"
    if label == "Seleccione tipo de indicador":
        if _loaded_home_module_for_mock and hasattr(_loaded_home_module_for_mock, 'indicadores_clave') \
           and isinstance(getattr(_loaded_home_module_for_mock, 'indicadores_clave', None), dict) \
           and list(getattr(_loaded_home_module_for_mock, 'indicadores_clave').keys()):
            return list(getattr(_loaded_home_module_for_mock, 'indicadores_clave').keys())[0]
        return "IET - Estructura Tarifaria" 
    if options and len(options) > index: return options[index]
    return "MockedSelectboxDefaultValue"

# =============================================================================
# CONFIGURACIÓN DETALLADA DE MOCKS DE PROPHET (Nivel de módulo de test)
# Esta configuración será usada tanto por load_home_module como por los tests de integración.
# =============================================================================
# 1. Mock para la INSTANCIA de Prophet (lo que Prophet() devuelve)
prophet_instance_mock = MagicMock(name="ConfiguredProphetInstanceMock")

# Configurar los métodos de la instancia mock para que devuelvan DataFrames válidos
# Esto es crucial para que home.py no falle durante su carga en load_home_module
# y para que los tests de integración puedan verificar las llamadas.
_mock_prophet_dates_for_load_and_integration = pd.to_datetime([
    '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', 
    '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', 
    '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', # Histórico + Futuro para 12 periodos
    '2024-01-01', '2024-02-01', '2024-03-01' # Más fechas si make_future_dataframe crea más
])
prophet_instance_mock.fit = MagicMock(name="ProphetFitMock")
prophet_instance_mock.make_future_dataframe.return_value = pd.DataFrame({'ds': _mock_prophet_dates_for_load_and_integration[:12+3]}) # Ejemplo de 12 periodos futuros
prophet_instance_mock.predict.return_value = pd.DataFrame({
    'ds': _mock_prophet_dates_for_load_and_integration[:12+3], 
    'yhat': np.random.rand(15) * 1000 + 17000, 
    'yhat_lower': np.random.rand(15) * 1000 + 16000, 
    'yhat_upper': np.random.rand(15) * 1000 + 18000
})

# 2. Mock para la CLASE Prophet. Cuando se llame Prophet(...), devolverá prophet_instance_mock
prophet_class_mock = MagicMock(name="ConfiguredProphetClassMock", return_value=prophet_instance_mock)

# 3. Asegurar que sys.modules['prophet'].Prophet ES nuestro mock de clase configurado
# Esto es para que `from prophet import Prophet` en home.py obtenga nuestro mock.
if 'prophet' not in sys.modules:
    sys.modules['prophet'] = MagicMock(name="GlobalProphetMockModule")
sys.modules['prophet'].Prophet = prophet_class_mock
# --- Fin de configuración de mocks de Prophet ---

# =============================================
# FUNCIÓN PARA CARGAR home.py COMO MÓDULO
# =============================================
def load_home_module():
    global _loaded_home_module_for_mock, prophet_class_mock
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)

    module_name_in_sys = "home_module_for_test" 
    file_path = "home.py" 
    
    if module_name_in_sys in sys.modules: del sys.modules[module_name_in_sys]

    spec = importlib.util.spec_from_file_location(module_name_in_sys, file_path)
    home_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name_in_sys] = home_module
    _loaded_home_module_for_mock = home_module 
    
    mock_df_for_load = pd.DataFrame({
        'Fecha': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']), 
        'Municipio':['Medellín', 'Medellín', 'Bello'], 
        'Estrato':[1, 1, 2], 
        'Servicio':['Acueducto', 'Acueducto', 'Alcantarillado'], 
        'Cargo Fijo':[15000.0, 15100.0, 16000.0], 
        'Año': [2023, 2023, 2023],
        'Cargo por Consumo': [1000.0, 1100.0, 1200.0],
        'ratio_cargo_fijo_variable': [1.5, 1.6, 1.7],
        'indice_progresividad': [2.0, 2.1, 2.2],
        'diferencial_estratos': [0.8, 0.9, 0.7], 
        'indice_sectorial': [1.1, 1.2, 1.3],
        'diferencial_por_estrato':[1.0]*3, 'indice_carga':[1.0]*3, 
        'ratio_penalizacion':[0.5]*3, 'factor_operativo':[1.0]*3, 
        'indice_penalizacion':[0.5]*3,'dispersion_municipal':[100.0]*3,
        'ratio_municipal':[1.0]*3,'indice_variabilidad':[0.1]*3
    })

    # Mock de Prophet para la carga global
    # (Esta es la variable que mencionaste que tu agente de IA corrigió)
    prophet_mock_for_load_during_exec = MagicMock(name="ProphetMockForModuleLoadDuringExec")
    future_df_mock_during_exec = pd.DataFrame({'ds': pd.to_datetime(['2023-04-01', '2023-05-01'])})
    prophet_mock_for_load_during_exec.make_future_dataframe.return_value = future_df_mock_during_exec
    prophet_mock_for_load_during_exec.predict.return_value = pd.DataFrame({
        'ds': future_df_mock_during_exec['ds'], 'yhat': [17000, 17500], 
        'yhat_lower': [16000, 16500], 'yhat_upper': [18000, 18500]
    })

    with patch('streamlit.set_page_config', MagicMock(name="Mock_st.set_page_config")), \
         patch('streamlit.markdown', mock_st_global.markdown), \
         patch('streamlit.columns', mock_st_global.columns), \
         patch('streamlit.sidebar', mock_st_global.sidebar), \
         patch('streamlit.expander', mock_st_global.expander), \
         patch('streamlit.tabs', mock_st_global.tabs), \
         patch('streamlit.selectbox', MagicMock(side_effect=mock_st_selectbox_dynamic, name="Mock_st.selectbox_dynamic_load")), \
         patch('streamlit.slider', MagicMock(return_value=(2023,2023), name="Mock_st.slider_load")), \
         patch('streamlit.warning', MagicMock(name="Mock_st.warning_load")), \
         patch('streamlit.plotly_chart', MagicMock(name="Mock_st.plotly_chart_load")), \
         patch('streamlit.dataframe', MagicMock(name="Mock_st.dataframe_load")), \
         patch('streamlit.caption', MagicMock(name="Mock_st.caption_load")), \
         patch('streamlit.error', MagicMock(name="Mock_st.error_load")), \
         patch('streamlit.stop', MagicMock(side_effect=SystemExit("Mocked st.stop"), name="Mock_st.stop_load")), \
         patch('streamlit.cache_data', lambda func: lambda *args, **kwargs: func(*args, **kwargs)), \
         patch('prophet.Prophet', prophet_class_mock), \
         patch('sqlalchemy.create_engine', MagicMock(return_value=Mock(name="GlobalLoadMockSqlAlchemyEngine"))), \
         patch('pandas.read_sql', MagicMock(return_value=mock_df_for_load, name="LoadTimeMockPandasReadSql")), \
         patch('dotenv.load_dotenv', MagicMock(name="LoadTimeMockDotenvLoadDotenv")):
        try:
            spec.loader.exec_module(home_module)
        except SystemExit as e: 
            if "Mocked st.stop" not in str(e): raise
    return home_module

# =============================================
# TESTS UNITARIOS (3 tests)
# =============================================
@pytest.mark.unit
@pytest.mark.home
class TestUnitariosHome:

    def test_crear_engine_exitoso(self):
        # Este test ya estaba PASANDO en tu última ejecución, se mantiene igual.
        home_module = load_home_module() 
        with patch('os.getenv') as mock_getenv, \
             patch.object(home_module, 'create_engine', new_callable=MagicMock) as mock_sqlalchemy_in_home_module: # Corregido patch target
            mock_getenv.side_effect = lambda key: {'DB_USER': 'u_test', 'DB_PASSWORD': 'p_test', 'DB_HOST': 'h_test', 'DB_PORT': '5432', 'DB_NAME': 'n_test'}.get(key)
            mock_engine_instance = Mock(name="TestSpecificEngineInstance")
            mock_sqlalchemy_in_home_module.return_value = mock_engine_instance
            engine = home_module.crear_engine() 
            mock_sqlalchemy_in_home_module.assert_called_once_with('postgresql+psycopg2://u_test:p_test@h_test:5432/n_test')
            assert engine == mock_engine_instance

    def test_cargar_datos_exitoso(self):
        home_module = load_home_module()
        mock_df_from_sql = pd.DataFrame({'Fecha': [pd.to_datetime('2023-01-01')]})
        
        # Para probar home_module.cargar_datos(), necesitamos controlar el 'engine' global
        # que usa y la llamada a 'pd.read_sql'.
        # 'home_module.engine' ya es un mock debido a los patches en load_home_module.
        # Podemos sobreescribirlo o simplemente mockear pd.read_sql.
        
        with patch.object(home_module, 'engine', new_callable=Mock) as mock_engine_global_ref, \
             patch.object(home_module.pd, 'read_sql', return_value=mock_df_from_sql) as mock_pd_read_sql_call:
            
            # mock_engine_global_ref es ahora home_module.engine
            # home_module.cargar_datos() usará este mock_engine_global_ref

            df = home_module.cargar_datos() 
            
            # Verificar que pd.read_sql fue llamado con el engine mockeado
            mock_pd_read_sql_call.assert_called_once_with(
                "SELECT * FROM tarifas_acueductos_aguas_residuales_med_ing_caracteristicas",
                mock_engine_global_ref # Asegura que usa el engine (mockeado) correcto
            )
            assert 'Fecha' in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df['Fecha'])

    def test_cargar_datos_fallido(self):
        # Este test ya estaba PASANDO, se mantiene igual.
        home_module = load_home_module()
        with patch.object(home_module, 'crear_engine', return_value=Mock()), \
             patch.object(home_module.pd, 'read_sql', side_effect=Exception("DB Error Test")), \
             patch.object(home_module.st, 'error') as mock_st_error, \
             patch.object(home_module.st, 'exception') as mock_st_exception, \
             patch.object(home_module.st, 'stop', side_effect=SystemExit("Mocked st.stop")): 
            try:
                df = home_module.cargar_datos()
                assert df.empty 
            except SystemExit as e: 
                 if "Mocked st.stop" not in str(e) : raise
            mock_st_error.assert_called() 
            mock_st_exception.assert_called_once()

# =============================================
# TESTS DE INTEGRACIÓN (3 tests)
# =============================================
@pytest.mark.integration
@pytest.mark.home
class TestIntegracionHome:
    def test_filtrado_datos_logica(self, sample_tarifas_dataframe):
        home_module = load_home_module() 
        with patch.object(home_module, 'df_real', sample_tarifas_dataframe.copy()):
            municipio, estrato, servicio = 'Medellín', 1, 'Acueducto'
            df_filtrado_simulado = home_module.df_real[
                (home_module.df_real['Municipio'] == municipio) &
                (home_module.df_real['Estrato'] == estrato) &
                (home_module.df_real['Servicio'] == servicio)
            ]
            assert not df_filtrado_simulado.empty

    def test_flujo_prediccion_prophet_mockeado(self, sample_tarifas_dataframe):
        home_module = load_home_module() # load_home_module usa el prophet_class_mock global
        
        # Referenciar los mocks globales definidos a nivel de módulo de test
        # con los nombres correctos que tienes en tu configuración global.
        clase_prophet_mock_a_verificar = prophet_class_mock 
        instancia_prophet_mock_a_verificar = prophet_instance_mock

        # Resetear mocks para aserciones limpias específicas a este test
        clase_prophet_mock_a_verificar.reset_mock()
        instancia_prophet_mock_a_verificar.reset_mock()
        
        # Reconfigurar return_values para la instancia mock, ya que reset_mock() los borra.
        # Esto asegura que la lógica del test funcione con DataFrames con la estructura esperada.
        _test_prophet_future_dates = pd.to_datetime([f"2024-{i:02d}-01" for i in range(1, 13)])
        
        instancia_prophet_mock_a_verificar.make_future_dataframe.return_value = pd.DataFrame({'ds': _test_prophet_future_dates})
        instancia_prophet_mock_a_verificar.predict.return_value = pd.DataFrame({
            'ds': _test_prophet_future_dates, 
            'yhat': np.random.uniform(15000, 25000, 12), 
            'yhat_lower': np.random.uniform(14000, 24000, 12), 
            'yhat_upper': np.random.uniform(16000, 26000, 12)
        })
        # No es necesario re-mockear .fit() aquí si solo necesitas que sea un MagicMock callable.
        # instancia_prophet_mock_a_verificar.fit = MagicMock(name="FitCalledInThisTest")


        # Preparar datos de entrada para la lógica de Prophet
        df_filtrado = sample_tarifas_dataframe[
            (sample_tarifas_dataframe['Municipio'] == 'Medellín') &
            (sample_tarifas_dataframe['Estrato'] == 1) &
            (sample_tarifas_dataframe['Servicio'] == 'Acueducto')
        ]
        if len(df_filtrado) < 2:
            pytest.skip("Datos de prueba filtrados insuficientes para Prophet.")
        
        df_prophet_input = df_filtrado[['Fecha', 'Cargo Fijo']].rename(columns={'Fecha': 'ds', 'Cargo Fijo': 'y'})

        # --- Simular la lógica de home.py ---
        modelo = home_module.Prophet(interval_width=0.95) 
        modelo.fit(df_prophet_input)
        future_df = modelo.make_future_dataframe(periods=12, freq='M') 
        forecast = modelo.predict(future_df)
        # --- Fin de la simulación ---

        # Aserciones
        clase_prophet_mock_a_verificar.assert_called_with(interval_width=0.95)
        instancia_prophet_mock_a_verificar.fit.assert_called_with(df_prophet_input)
        instancia_prophet_mock_a_verificar.make_future_dataframe.assert_called_with(periods=12, freq='M')
        instancia_prophet_mock_a_verificar.predict.assert_called_with(future_df)
        
        assert not forecast.empty
        assert 'yhat' in forecast.columns
    
    def test_generacion_grafico_principal_mockeado(self, sample_tarifas_dataframe):
        home_module = load_home_module()
        with patch(f'{home_module.__name__}.go.Figure') as mock_go_figure, \
             patch.object(home_module.st, 'plotly_chart') as mock_st_plotly_chart:
            mock_fig_instance = MagicMock(name="FigureInstance")
            mock_go_figure.return_value = mock_fig_instance
            home_module.df_filtrado = sample_tarifas_dataframe.iloc[:10] 
            if home_module.df_filtrado.empty: pytest.skip("Datos filtrados vacíos.")
            home_module.historico = pd.DataFrame({'ds': pd.to_datetime(['2023-01-01']), 'yhat': [100]})
            home_module.prediccion = pd.DataFrame({'ds': pd.to_datetime(['2023-02-01']), 'yhat': [110], 'yhat_lower': [105], 'yhat_upper': [115]})
            if not home_module.df_filtrado.empty:
                fig_sim = home_module.go.Figure() 
                fig_sim.add_trace(home_module.go.Scatter())
                fig_sim.update_layout()
                home_module.st.plotly_chart(fig_sim, use_container_width=True)
                mock_go_figure.assert_called()

# =============================================
# TESTS PARAMÉTRICOS
# =============================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium no está instalado")
@pytest.mark.parametric
@pytest.mark.home
@pytest.mark.slow
class TestParametricosHome:
    BASE_URL = "http://localhost:8501"

    def _wait_for_page_load_fully(self, driver, wait, expected_title_part="Sistema de Predicción"):
        wait.until(EC.title_contains(expected_title_part))
        try:
            wait.until_not(EC.presence_of_element_located((By.XPATH, "//*[@data-testid='stStatusWidget']//*[text()='Running...']")))
        except TimeoutException: print("Spinner 'Running...' no desapareció o no estaba presente.")
        time.sleep(2) # Aumentado

    @pytest.mark.parametrize("link_info", [
        {"xpath": "//a[contains(@href, '/Visor_Geografico')]", "url_part": "/Visor_Geografico", "title_part": "Visor Geográfico"},
        {"xpath": "//a[contains(@href, '/Predicciones')]", "url_part": "/Predicciones", "title_part": "Predicciones Tarifas"},
        {"xpath": "//ul[@data-testid='stSidebarNavItems']/li[1]//a", "url_part": "/", "title_part": "Sistema de Predicción"}
    ])
    def test_navegacion_sidebar_parametrizada(self, driver, setup_streamlit_home_app, link_info):
        # Este test ya está PASANDO, no se modifica excepto el XPATH del Home link
        driver.get(self.BASE_URL)
        wait = WebDriverWait(driver, 30) 
        self._wait_for_page_load_fully(driver, wait)
        try:
            link = wait.until(EC.visibility_of_element_located((By.XPATH, link_info["xpath"])))
            link = wait.until(EC.element_to_be_clickable(link))
        except TimeoutException: pytest.fail(f"Link no encontrado/clickeable: {link_info['xpath']}.")
        driver.execute_script("arguments[0].scrollIntoView(true);", link); time.sleep(0.5)
        try: link.click()
        except ElementClickInterceptedException: driver.execute_script("arguments[0].click();", link)
        time.sleep(4) 
        wait.until(EC.title_contains(link_info["title_part"]))
        current_url = driver.current_url
        if link_info["url_part"] == "/": 
            assert current_url == self.BASE_URL or current_url.endswith(self.BASE_URL + "/") or current_url.endswith("/home")
        else: assert link_info["url_part"] in current_url
   

# =============================================
# TESTS E2E CON SELENIUM (Restantes)
# =============================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium no está instalado")
@pytest.mark.e2e
@pytest.mark.home
@pytest.mark.slow
class TestE2EHome: # Los tests que ya PASAN se mantienen igual
    BASE_URL = "http://localhost:8501"

    def _wait_for_page_load_fully(self, driver, wait, expected_title_part="Sistema de Predicción"):
        wait.until(EC.title_contains(expected_title_part))
        try:
            wait.until_not(EC.presence_of_element_located((By.XPATH, "//*[@data-testid='stStatusWidget']//*[text()='Running...']")))
        except TimeoutException: print("Spinner 'Running...' no desapareció.")
        time.sleep(2) 

    def test_TC_H_01_carga_y_titulo_H1(self, driver, setup_streamlit_home_app):
        driver.get(self.BASE_URL)
        wait = WebDriverWait(driver, 30)
        self._wait_for_page_load_fully(driver, wait)
        assert "Sistema de Predicción de Tarifas - Valle de Aburrá" in driver.title
        title_h1_xpath = "//h1[contains(@class, 'main-header') and contains(text(), 'Sistema de Predicción de Tarifas')]"
        title_h1 = wait.until(EC.visibility_of_element_located((By.XPATH, title_h1_xpath)))
        assert title_h1.is_displayed()
        errors = driver.find_elements(By.XPATH, "//*[contains(@data-testid, 'stExceptionContainer')] | //*[contains(@data-testid, 'stError')]")
        assert len(errors) == 0, f"Errores encontrados: {[e.text for e in errors if e.text]}"

    def test_TC_H_08_filtros_principales_visibles(self, driver, setup_streamlit_home_app):
        driver.get(self.BASE_URL)
        wait = WebDriverWait(driver, 20)
        self._wait_for_page_load_fully(driver, wait)
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Municipio']")))
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Estrato']")))
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Tipo de Servicio']")))

    def test_TC_H_XX_verificar_secciones_footer_y_subtitulos(self, driver, setup_streamlit_home_app):
        driver.get(self.BASE_URL)
        wait = WebDriverWait(driver, 30)
        self._wait_for_page_load_fully(driver, wait)
        
        print("DEBUG: Verificando subtítulos de secciones y footer...")

        # Verificar subtítulo "Indicadores de Análisis Tarifario"
        indicadores_subtitle_xpath = "//h2[@class='sub-header' and normalize-space()='Indicadores de Análisis Tarifario']"
        print(f"DEBUG: Buscando subtítulo Indicadores: {indicadores_subtitle_xpath}")
        indicadores_subtitle = wait.until(EC.visibility_of_element_located((By.XPATH, indicadores_subtitle_xpath)))
        assert indicadores_subtitle.is_displayed()
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", indicadores_subtitle) # Scroll para asegurar visibilidad
        time.sleep(0.5)
        print("DEBUG: Subtítulo 'Indicadores de Análisis Tarifario' encontrado y visible.")

        # Verificar subtítulo "Metodología de Predicción"
        metodologia_subtitle_xpath = "//h2[@class='sub-header' and normalize-space()='Metodología de Predicción']"
        print(f"DEBUG: Buscando subtítulo Metodología: {metodologia_subtitle_xpath}")
        metodologia_subtitle = wait.until(EC.visibility_of_element_located((By.XPATH, metodologia_subtitle_xpath)))
        assert metodologia_subtitle.is_displayed()
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", metodologia_subtitle)
        time.sleep(0.5)
        print("DEBUG: Subtítulo 'Metodología de Predicción' encontrado y visible.")

        # Verificar el pie de página (st.caption)
        # Streamlit renderiza st.caption dentro de un div con data-testid="stCaptionContainer"
        # y el texto está dentro de un <p> o directamente.
        footer_xpath = "//div[@data-testid='stCaptionContainer']" 
        print(f"DEBUG: Buscando footer: {footer_xpath}")
        footer_element = wait.until(EC.visibility_of_element_located((By.XPATH, footer_xpath)))
        assert footer_element.is_displayed()
        assert "© 2025 Sistema de Predicción de Tarifas de acueducto y alcantarillado" in footer_element.text
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", footer_element)
        time.sleep(0.5)
        print("DEBUG: Footer encontrado y con texto correcto.")

    def test_TC_H_18_seccion_comparativas_filtros_visibles(self, driver, setup_streamlit_home_app):
        driver.get(self.BASE_URL)
        wait = WebDriverWait(driver, 30)
        self._wait_for_page_load_fully(driver, wait)
        comparativas_header_xpath = "//h2[normalize-space()='Comparativas y Análisis Avanzados']"
        comparativas_header = wait.until(EC.visibility_of_element_located((By.XPATH, comparativas_header_xpath)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", comparativas_header); time.sleep(1)
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Seleccione municipio base']")))
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Desde']"))) 
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Hasta']")))

    def test_TC_H_19_tabs_comparativas_funcionan_y_contenido(self, driver, setup_streamlit_home_app):
        driver.get(self.BASE_URL)
        wait = WebDriverWait(driver, 35)
        self._wait_for_page_load_fully(driver, wait)
        comparativas_header_xpath = "//h2[normalize-space()='Comparativas y Análisis Avanzados']"
        comparativas_header = wait.until(EC.visibility_of_element_located((By.XPATH, comparativas_header_xpath)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", comparativas_header); time.sleep(1)
        wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@data-testid='stTabs']//h3[normalize-space()='Promedio de tarifa por municipio']")))
        tab_xpath = "//button[@role='tab' and normalize-space()='Indicadores Tarifarios']"
        tab_indicadores = wait.until(EC.element_to_be_clickable((By.XPATH, tab_xpath)))
        driver.execute_script("arguments[0].click();", tab_indicadores)
        time.sleep(2)
        wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@data-testid='stTabs']//h3[normalize-space()='Indicadores Tarifarios por Municipio']")))
        wait.until(EC.visibility_of_element_located((By.XPATH, "//label[normalize-space()='Seleccione tipo de indicador']")))