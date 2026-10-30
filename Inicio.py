import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import platform
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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
    .stApp {
        background-color: #F4FBF7;
        color: #1B3B2B;
    }
    h1, h2, h3 {
        color: #0F2A1D !important;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0F2A1D;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
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
    .result-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #059669;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .stAlert {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Panel de Control y Propiedades del Tablero en el Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/bot.png", width=70)
    st.subheader("🛠️ Entorno Técnico")
    st.markdown(f"**Python:** `{platform.python_version()}`")
    st.markdown(f"**Motor:** `Scikit-Learn (MLP)`")
    
    st.markdown("---")
    st.subheader("🎨 Configuración del Tablero")
    
    stroke_width = st.slider("Grosor del Pincel", min_value=10, max_value=40, value=22, step=2)
    stroke_color = st.color_picker("Color del Pincel", "#FFFFFF")
    bg_color = st.color_picker("Color de Fondo", "#000000")
    canvas_size = st.slider("Tamaño del Lienzo (px)", min_value=200, max_value=400, value=300, step=50)
    
    st.markdown("---")
    st.info("💡 **Tip:** Dibuja un número claro en el centro del recuadro para obtener una mejor predicción.")

# Título y presentación
st.title("✍️ Detección de Dígitos Escritos a Mano")
st.markdown("*Modelo de Red Neuronal Ligero optimizado para correr de forma fluida en Streamlit Cloud.*")

st.markdown("---")

# Cargar y entrenar un modelo optimizado en caché para evitar demoras
@st.cache_resource
def load_optimized_model():
    """Carga el dataset de dígitos y entrena un clasificador neuronal rápido"""
    digits = load_digits()
    X, y = digits.data, digits.target
    # Normalizar datos
    X = X / 16.0
    # Entrenar un Perceptrón Multicapa (MLP) optimizado para velocidad y precisión
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42, alpha=1e-4)
    clf.fit(X, y)
    return clf

with st.spinner("Inicializando modelo de inteligencia artificial..."):
    model = load_optimized_model()

st.markdown("### 🎨 Lienzo de Dibujo Interactivo")

col_canvas, col_controls = st.columns([1.2, 1], gap="large")

with col_canvas:
    st.write("Dibuja un dígito del **0 al 9**:")
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
    st.markdown("Presiona **Predecir Dígito** para analizar tu trazo o **Limpiar** para reiniciar el lienzo.")
    
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
            with st.spinner("🤔 Analizando trazo..."):
                # Procesar imagen a escala de grises
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")
                
                # Redimensionar a 8x8 píxeles (que es el formato nativo optimizado para scikit-learn digits)
                image_resized = image.resize((8, 8), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized, dtype=float)
                
                # Invertir colores si el fondo es negro para estandarizar con el dataset
                if bg_color == "#000000":
                    img_array = 255.0 - img_array
                    
                # Escalar los valores de píxeles al rango de entrenamiento [0, 16]
                img_array = (img_array / 255.0) * 16.0
                
                # Aplanar la matriz a un vector de 64 características
                img_vector = img_array.reshape(1, -1)
                
                # Predicción con el modelo
                prediction_proba = model.predict_proba(img_vector)[0]
                digit = int(np.argmax(prediction_proba))
                confidence = float(np.max(prediction_proba) * 100)
                alternative = int(np.argsort(prediction_proba)[-2])
            
            # Resultados principales en una tarjeta estilizada
            st.markdown("---")
            st.markdown(f"""
                <div class="result-box">
                    <h2>🎯 Dígito Detectado: <span style="color: #059669;">{digit}</span></h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Métricas en columnas
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(label="Predicción Principal", value=str(digit))
            with col_m2:
                st.metric(label="Nivel de Confianza", value=f"{confidence:.1f}%")
            with col_m3:
                st.metric(label="Segunda Opción", value=str(alternative))
            
            st.markdown("---")
            
            # Visualización del procesamiento
            st.write("### 🖼️ Transformación de la Imagen")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.write("**Tu dibujo original:**")
                st.image(canvas_result.image_data, width=180)
            
            with col_img2:
                st.write("**Entrada procesada (8x8):**")
                st.image(image_resized.resize((150, 150), Image.Resampling.NEAREST))
            
            # Gráfico de probabilidades
            st.write("### 📊 Distribución de Probabilidades")
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction_proba * 100
            })
            st.bar_chart(prob_df.set_index('Dígito'), color="#059669")
            
            with st.expander("🔍 Ver desglose detallado por dígito"):
                for i, prob in enumerate(prediction_proba):
                    emoji = "🎯 (Seleccionado)" if i == digit else ""
                    st.write(f"**Dígito {i}**: {prob*100:.2f}% {emoji}")
    else:
        st.warning("⚠️ No se detectó información en el lienzo.", icon="⚠️")
