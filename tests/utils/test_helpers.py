"""
Funciones auxiliares extraídas para testing
Estas funciones son necesarias para hacer los módulos principales más testables
"""

import pandas as pd
import numpy as np
import unidecode
import geopandas as gpd
import folium
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ============================================
# FUNCIONES DEL VISOR GEOGRÁFICO
# ============================================

def normalizar_nombre(texto):
    """
    Normaliza nombres de municipios para comparación
    
    Args:
        texto: Nombre a normalizar
        
    Returns:
        str: Nombre normalizado en minúsculas sin espacios ni acentos
    """
    if pd.isna(texto):
        return texto
    texto = str(texto).lower()
    texto = unidecode.unidecode(texto)
    texto = texto.replace(' ', '')
    return texto

def get_base_map(selection):
    """
    Obtiene la URL del mapa base según la selección
    
    Args:
        selection: Tipo de mapa seleccionado
        
    Returns:
        str: URL o identificador del mapa base
    """
    if selection == "OpenStreetMap":
        return "OpenStreetMap"
    elif selection == "Cartografía":
        return "CartoDB positron"
    elif selection == "Satélite":
        return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    elif selection == "Terreno":
        return "https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg"
    else:
        return "OpenStreetMap"

def get_attribution(selection):
    """
    Obtiene la atribución correspondiente al tipo de mapa
    
    Args:
        selection: Tipo de mapa seleccionado
        
    Returns:
        str: Texto de atribución del mapa
    """
    if selection == "Satélite":
        return "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    elif selection == "Terreno":
        return "Map tiles by <a href='http://stamen.com'>Stamen Design</a>, <a href='http://creativecommons.org/licenses/by/3.0'>CC BY 3.0</a> &mdash; Map data &copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
    else:
        return "© OpenStreetMap contributors"

# Constantes del visor geográfico
CENTRO_VALLE_ABURRA = [6.25184, -75.56359]

INDICADORES = {
    "Diferencial por Estrato": "diferencial_por_estrato",
    "Índice de Carga": "indice_carga",
    "Ratio de Penalización": "ratio_penalizacion",
    "Factor Operativo": "factor_operativo",
    "Índice de Penalización": "indice_penalizacion",
    "Dispersión Municipal": "dispersion_municipal",
    "Ratio Municipal": "ratio_municipal",
    "Índice de Variabilidad": "indice_variabilidad"
}

ESCALAS_COLORES = {
    "Diferencial por Estrato": ['#e5f5e0', '#a1d99b', '#31a354', '#006d2c', '#00441b'],
    "Índice de Carga": ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#2171b5'],
    "Ratio de Penalización": ['#fee6ce', '#fdd0a2', '#fdae6b', '#f16913', '#d94801'],
    "Factor Operativo": ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#6a51a3'],
    "Índice de Penalización": ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
    "Dispersión Municipal": ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#4292c6'],
    "Ratio Municipal": ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#41ab5d'],
    "Índice de Variabilidad": ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59']
}

def cargar_datos():
    """
    Carga datos de shapefiles y CSV para el visor geográfico
    
    Returns:
        tuple: (GeoDataFrame de municipios, DataFrame de tarifas)
    """
    try:
        # Cargar shapefile de municipios
        gdf_municipios = gpd.read_file('data/shp/municipios.shp')
        
        # Cargar datos de tarifas e indicadores
        df_tarifas = pd.read_csv('data/tarifas_con_indicadores.csv')
        
        # Normalizar nombres de municipios
        gdf_municipios['MpNombre_norm'] = gdf_municipios['MpNombre'].apply(normalizar_nombre)
        df_tarifas['Municipio_norm'] = df_tarifas['Municipio'].apply(normalizar_nombre)
        
        return gdf_municipios, df_tarifas
    except Exception as e:
        return None, None

def crear_mapa(geojson_data, columna_indicador, indicador_seleccionado, mapa_base, vmin, vmax, municipio_seleccionado=None):
    """
    Crea un mapa Folium con los datos geográficos y indicadores
    
    Args:
        geojson_data: Datos GeoJSON de los municipios
        columna_indicador: Nombre de la columna del indicador
        indicador_seleccionado: Nombre del indicador seleccionado
        mapa_base: Tipo de mapa base
        vmin: Valor mínimo para la escala de colores
        vmax: Valor máximo para la escala de colores
        municipio_seleccionado: Municipio a resaltar (opcional)
        
    Returns:
        folium.Map: Mapa configurado
    """
    m = folium.Map(
        location=CENTRO_VALLE_ABURRA,
        zoom_start=10,
        tiles=None
    )
    
    # Añadir capa base
    folium.TileLayer(
        tiles=get_base_map(mapa_base),
        attr=get_attribution(mapa_base),
        name=mapa_base
    ).add_to(m)
    
    # Definir la escala de colores
    colormap = folium.LinearColormap(
        colors=ESCALAS_COLORES[indicador_seleccionado],
        vmin=vmin,
        vmax=vmax,
        caption=f'Valor de {indicador_seleccionado}'
    )
    
    # Función para el estilo de cada polígono
    def style_function(feature):
        valor = feature['properties'].get(columna_indicador)
        nombre = feature['properties'].get('MpNombre')
        
        if municipio_seleccionado and nombre == municipio_seleccionado:
            return {
                'fillColor': colormap(valor) if not pd.isna(valor) else '#808080',
                'color': '#FF0000',
                'fillOpacity': 0.9,
                'weight': 3
            }
        
        return {
            'fillColor': '#808080' if pd.isna(valor) else colormap(valor),
            'color': '#0D47A1',
            'fillOpacity': 0.7,
            'weight': 1
        }
    
    # Añadir capa de municipios al mapa
    folium.GeoJson(
        geojson_data,
        name='Municipios',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['MpNombre', columna_indicador],
            aliases=['Municipio:', f'{indicador_seleccionado}:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)
    
    # Añadir la leyenda de colores
    colormap.add_to(m)
    
    # Añadir control de capas
    folium.LayerControl().add_to(m)
    
    return m

# ============================================
# FUNCIONES DEL MÓDULO DE PREDICCIONES
# ============================================

def hex_to_rgba(hex_color, alpha=0.2):
    """
    Convierte un color hexadecimal a formato RGBA
    
    Args:
        hex_color: Color en formato hexadecimal
        alpha: Valor de transparencia (0-1)
        
    Returns:
        str: Color en formato rgba(r,g,b,alpha)
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def crear_engine():
    """
    Crea el engine de conexión a la base de datos PostgreSQL
    
    Returns:
        sqlalchemy.Engine: Engine de conexión configurado
        
    Raises:
        Exception: Si hay error en la conexión
    """
    try:
        load_dotenv()
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")

        return create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    except Exception as e:
        import streamlit as st
        st.error("Error al conectar con la base de datos.")
        st.stop()

def cargar_datos_bd(engine):
    """
    Carga datos desde la base de datos PostgreSQL
    
    Args:
        engine: Engine de conexión a la base de datos
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        query = "SELECT * FROM tarifas_acueductos_aguas_residuales_med_ing_caracteristicas"
        df = pd.read_sql(query, engine)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        return df
    except Exception as e:
        import streamlit as st
        st.error("Error al cargar los datos desde la base de datos.")
        st.exception(e)
        return pd.DataFrame()

def validar_columnas_requeridas(df, columnas_requeridas):
    """
    Valida que un DataFrame contenga las columnas requeridas
    
    Args:
        df: DataFrame a validar
        columnas_requeridas: Lista de columnas que deben estar presentes
        
    Returns:
        bool: True si todas las columnas están presentes, False en caso contrario
    """
    return all(col in df.columns for col in columnas_requeridas)

def filtrar_datos_por_parametros(df, municipio, estrato, servicio):
    """
    Filtra el DataFrame por los parámetros especificados
    
    Args:
        df: DataFrame a filtrar
        municipio: Nombre del municipio
        estrato: Número del estrato
        servicio: Tipo de servicio
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    return df[
        (df['Municipio'] == municipio) &
        (df['Estrato'] == estrato) &
        (df['Servicio'] == servicio)
    ].sort_values('Fecha')

def preparar_serie_temporal(df_filtrado):
    """
    Prepara los datos para modelos de series temporales
    
    Args:
        df_filtrado: DataFrame filtrado con datos de tarifas
        
    Returns:
        pd.DataFrame: DataFrame con columnas 'ds' (fecha) y 'y' (valor)
    """
    return df_filtrado[['Fecha', 'Cargo Fijo']].rename(columns={'Fecha': 'ds', 'Cargo Fijo': 'y'})

def calcular_metricas_modelo(y_true, y_pred):
    """
    Calcula métricas de evaluación para modelos de predicción
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        dict: Diccionario con métricas MAPE y RMSE
    """
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'MAPE': mape,
        'RMSE': rmse
    }

# Información de modelos para la interfaz
MODELOS_INFO = {
    "ARIMA": {
        "descripcion": "Modelo estadístico que utiliza valores pasados y errores para predecir valores futuros. Captura patrones lineales y estacionalidad.",
        "fortalezas": ["Bueno para series temporales con patrones claros", "Funciona bien con datos estacionarios", "Ideal para predicciones a corto plazo"],
        "debilidades": ["Limitado para patrones no lineales", "Requiere datos estacionarios", "Sensible a valores atípicos"]
    },
    "Prophet": {
        "descripcion": "Desarrollado por Facebook, descompone la serie temporal en tendencia, estacionalidad y efectos de calendario.",
        "fortalezas": ["Maneja bien datos faltantes", "Detecta cambios en tendencias", "Robusto ante valores atípicos", "Incorpora efectos estacionales y de calendario"],
        "debilidades": ["Menos preciso en series muy irregulares", "Limitado en la incorporación de variables externas"]
    },
    "XGBoost": {
        "descripcion": "Implementación optimizada de árboles de decisión potenciados por gradiente. Incorpora características temporales y externas.",
        "fortalezas": ["Alta precisión predictiva", "Maneja grandes conjuntos de datos", "Puede incorporar variables explicativas adicionales"],
        "debilidades": ["Menos intuitivo para series temporales puras", "Requiere ingeniería de características temporal"]
    }
} 