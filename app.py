import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch

# Configuración inicial de la página
st.set_page_config(
    page_title="Telecom Customer Segmentation",
    page_icon="📱",
    layout="wide"
)

# Estilos CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        padding: 1rem;
        color: #2c3e50;
    }
    .highlight {
        color: #e74c3c;
        font-weight: bold;
    }
    .container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sub-header {
        color: #2980b9;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Cargar el modelo y los transformadores
@st.cache_resource
def load_model():
    model_data = joblib.load('optimized_nn_model.joblib')
    return model_data

def transform_features(data):
    """
    Aplica las transformaciones necesarias a los datos de entrada
    """
    df = data.copy()
    
    # Crear las características transformadas
    df['income_log'] = np.log1p(df['income'])
    df['age_scaled'] = df['age'] / 100
    df['tenure_scaled'] = df['tenure'] / 72
    df['employ_scaled'] = df['employ'] / 72
    df['retire_scaled'] = df['retire']
    
    return df

def show_welcome():
    # Título principal
    st.markdown("""
        <h1 class='main-title'>Sistema de Predicción de Segmentación de Clientes<br>
        <span class='highlight'>Telecomunicaciones</span></h1>
        """, unsafe_allow_html=True)

    # Descripción del Proyecto
    st.markdown("""
    <div class='container'>
    <p class='sub-header'>📋 Descripción del Proyecto</p>
    <p>Este sistema utiliza técnicas avanzadas de Machine Learning para predecir la categoría de servicio más adecuada 
    para clientes de telecomunicaciones, basándose en sus características demográficas y patrón de uso.</p>
    </div>
    """, unsafe_allow_html=True)

    # Categorías de Servicio y Variables de Predicción
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='container'>
        <p class='sub-header'>🎯 Categorías de Servicio</p>
        <ul>
            <li><strong>Basic Service (1):</strong> Servicios básicos de telecomunicaciones</li>
            <li><strong>E-Service (2):</strong> Servicios digitales y en línea</li>
            <li><strong>Plus Service (3):</strong> Servicios premium con beneficios adicionales</li>
            <li><strong>Total Service (4):</strong> Paquete completo de servicios</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='container'>
        <p class='sub-header'>🔍 Variables de Predicción</p>
        <ul>
            <li><strong>Demográficas:</strong> Región, edad, estado civil, género</li>
            <li><strong>Económicas:</strong> Ingresos, situación laboral</li>
            <li><strong>Servicio:</strong> Tiempo como cliente, dirección</li>
            <li><strong>Hogar:</strong> Número de residentes, estado de jubilación</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Instrucciones de Uso
    st.markdown("""
    <div class='container'>
    <p class='sub-header'>📝 Instrucciones de Uso</p>
    <ol>
        <li>Complete todos los campos del formulario con la información del cliente</li>
        <li>Asegúrese de que los valores ingresados sean correctos y estén dentro de los rangos esperados</li>
        <li>Presione el botón "Predecir" para obtener la categoría de servicio recomendada</li>
        <li>El sistema mostrará la predicción junto con las probabilidades para cada categoría</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Botón para comenzar
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Comenzar Predicción", help="Click para ir al formulario de predicción"):
        st.session_state['page'] = 'prediction'
        st.rerun()

def show_prediction_page():
    st.markdown("<h1 class='main-title'>Formulario de Predicción</h1>", unsafe_allow_html=True)
    
    # Crear el formulario
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            region = st.selectbox('Región', options=[1, 2, 3])
            tenure = st.number_input('Tiempo como cliente (años)', min_value=0, max_value=72)
            age = st.number_input('Edad', min_value=18, max_value=100)
            marital = st.selectbox('Estado civil', options=[0, 1], 
                                 format_func=lambda x: "Soltero" if x == 0 else "Casado")
            address = st.number_input('Años en la dirección actual', min_value=0)
            income = st.number_input('Ingreso (en miles)', min_value=0.0)
            
        with col2:
            ed = st.selectbox('Nivel educativo', options=[1, 2, 3, 4, 5])
            employ = st.number_input('Años empleado', min_value=0)
            retire = st.selectbox('Jubilado', options=[0, 1], 
                                format_func=lambda x: "No" if x == 0 else "Sí")
            gender = st.selectbox('Género', options=[0, 1], 
                                format_func=lambda x: "Femenino" if x == 0 else "Masculino")
            reside = st.number_input('Número de residentes', min_value=1)

        submitted = st.form_submit_button("Predecir")

    if submitted:
        try:
            # Cargar modelo y transformadores
            model_data = load_model()
            scaler = model_data['scaler']
            model_state = model_data['model_state']
            input_size = model_data['input_size']
            label_encoder = model_data['label_encoder']

            # Crear el modelo con la arquitectura correcta
            model = OptimizedNN(input_size)
            model.load_state_dict(model_state)
            model.eval()

            # Preparar los datos de entrada
            input_data = pd.DataFrame([[region, tenure, age, marital, address, income, 
                                      ed, employ, retire, gender, reside]], 
                                    columns=['region', 'tenure', 'age', 'marital', 'address', 
                                           'income', 'ed', 'employ', 'retire', 'gender', 'reside'])

            # Aplicar las transformaciones
            input_transformed = transform_features(input_data)
            
            # Verificar las columnas
            expected_columns = ['region', 'tenure', 'age', 'marital', 'address', 'income', 
                              'ed', 'employ', 'retire', 'gender', 'reside', 'income_log',
                              'age_scaled', 'tenure_scaled', 'employ_scaled', 'retire_scaled']
            
            input_transformed = input_transformed[expected_columns]
            
            # Aplicar el scaler
            input_scaled = scaler.transform(input_transformed)
            
            # Convertir a tensor y realizar predicción
            input_tensor = torch.FloatTensor(input_scaled)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
            
            # Obtener la categoría predicha
            predicted_category = label_encoder.inverse_transform(predicted.numpy())[0]
            
            # Mostrar resultado
            st.success(f'Categoría predicha: {predicted_category}')
            
            # Mostrar probabilidades
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
            st.write("Probabilidades por categoría:")
            for cat, prob in zip(label_encoder.classes_, probabilities):
                st.write(f"Categoría {cat}: {prob:.2%}")

        except Exception as e:
            st.error(f"Error durante la predicción: {str(e)}")
            st.write("Detalles del error para depuración:")
            st.write(e)

    # Botón para volver a la página de inicio
    if st.button("🏠 Volver al Inicio"):
        st.session_state['page'] = 'welcome'
        st.rerun()

class OptimizedNN(torch.nn.Module):
    def __init__(self, input_size):
        super(OptimizedNN, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(64, 4)
        )
        
    def forward(self, x):
        return self.layers(x)

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'

    if st.session_state['page'] == 'welcome':
        show_welcome()
    else:
        show_prediction_page()

if __name__ == "__main__":
    main()