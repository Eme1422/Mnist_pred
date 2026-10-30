import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import platform

# Configuración de la página (Debe ser el primer comando de Streamlit)
st.set_page_config(
    page_title="Detector de Dígitos MNIST",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados con paleta de tonos verdes
st.markdown("""
    <style>
    /* Fondo principal y color de texto general */
    .stApp {
        background-color: #F4FBF7;
        color: #1B3B2B;
    }
    
    /* Encabezados */
    h1, h2, h3 {
        color: #0F2A1D !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F2A1D;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    
    /* Botones principales */
    .stButton button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #047857 !important;
    }
    
    /* Tarjeta o contenedor para resultados */
    .result-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #059669;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Alertas y avisos */
    .stAlert {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Panel de Control y Propiedades del Tablero en el Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/bot.png", width=70)
    st.subheader("🛠️ Entorno Técnico")
    st.markdown(f"**TensorFlow:** `{tf.__version__}`")
    st.markdown(f"**Keras:** `{keras.__version__}`")
    st.markdown(f"**Python:** `{platform.python_version()}`")
    
    st.markdown("---")
    st.subheader("🎨 Configuración del Tablero")
    
    # Propiedades configurables del canvas
    stroke_width = st.slider("Grosor del Pincel", min_value=10, max_value=40, value=22, step=2)
    stroke_color = st.color_picker("Color del Pincel", "#FFFFFF")
    bg_color = st.color_picker("Color de Fondo", "#000000")
    canvas_size = st.slider("Tamaño del Lienzo (px)", min_value=200, max_value=400, value=300, step=50)
    
    st.markdown("---")
    st.info("💡 **Tip:** Dibuja un número claro en el centro del recuadro para obtener una mejor predicción.")

# Título y presentación
st.title("✍️ Detección de Dígitos Escritos a Mano")
st.markdown("*Modelo CNN entrenado con el dataset MNIST optimizado con Inteligencia Artificial.*")

st.markdown("---")

# Cargar modelo con caché optimizado para evitar retrasos
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras o .h5"""
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model
            except Exception as e:
                continue
    
    # Si no se encuentra el modelo
    st.error("❌ No se encontró ningún modelo de MNIST entrenado.")
    st.warning("""
    **Instrucciones para solucionarlo:**
    1. Entrena tu modelo CNN en Google Colab.
    2. Descarga el archivo `mnist_model.keras`.
    3. Súbelo a la raíz de tu repositorio/proyecto.
    4. Reinicia la aplicación.
    """)
    st.stop()

# Cargar modelo de forma eficiente
with st.spinner("Cargando modelo neuronal..."):
    model = load_mnist_model()

st.markdown("### 🎨 Lienzo de Dibujo Interactivo")

# Estructura principal en dos columnas
col_canvas, col_controls = st.columns([1.2, 1], gap="large")

with col_canvas:
    st.write(f"Dibuja un dígito del **0 al 9**:")
    # Se usan las propiedades personalizadas desde el panel lateral
    canvas_result = st_canvas(
        fill_color=bg_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )

with col_controls:
    st.write("### 🎮 Controles")
    st.markdown("Presiona **Predecir Dígito** para analizar tu trazo o **Limpiar** para reiniciar el lienzo
