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

# Mostrar información y versiones en el sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/bot.png", width=80)
    st.subheader("🛠️ Entorno Técnico")
    st.markdown(f"**TensorFlow:** `{tf.__version__}`")
    st.markdown(f"**Keras:** `{keras.__version__}`")
    st.markdown(f"**Python:** `{platform.python_version()}`")
    st.markdown("---")
    st.info("💡 **Tip:** Dibuja un número claro en el centro del recuadro para obtener una mejor predicción.")

# Título y presentación
st.title("✍️ Detección de Dígitos Escritos a Mano")
st.markdown("*Modelo CNN entrenado con el dataset MNIST optimizado con Inteligencia Artificial.*")

st.markdown("---")

# Cargar modelo con caché
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras o .h5"""
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                st.info(f"📂 Intentando cargar modelo desde: `{model_path}`")
                model = keras.models.load_model(model_path)
                st.success(f"✅ ¡Modelo cargado exitosamente!", icon="🎉")
                return model
            except Exception as e:
                st.error(f"❌ Error al cargar {model_path}: {str(e)}")
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

# Cargar modelo
model = load_mnist_model()

st.markdown("### 🎨 Lienzo de Dibujo")

# Estructura principal en dos columnas
col_canvas, col_controls = st.columns([1.2, 1], gap="large")

with col_canvas:
    st.write("Dibuja un dígito del **0 al 9** usando el ratón o tu dedo:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col_controls:
    st.write("### 🎮 Controles")
    st.markdown("Presiona **Predecir** para analizar tu trazo o **Limpiar** para reiniciar el lienzo.")
    
    predict_btn = st.button("🔍 **Predecir Dígito**", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Limpiar Lienzo", use_container_width=True)
    
    if clear_btn:
        st.rerun()

# Lógica de Predicción
if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ Por favor, dibuja un dígito en el lienzo antes de presionar predecir.", icon="⚠️")
        else:
            with st.spinner("🤔 Analizando trazo con la red neuronal..."):
                # Procesar imagen
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")
                
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predicción
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
