import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import os
from matplotlib.backends.backend_pdf import PdfPages

st.set_option('client.showErrorDetails', True)
st.set_page_config(layout="wide")
st.title("Visualizador de Grad-CAM - Alveolis Team")

# Cargar modelo base EfficientNetB0
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights=None)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(9, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

# Cargar pesos del modelo
model.load_weights("modelo_pesos.weights.h5")

# Cargar historial si existe
try:
    with open("history.json", "r") as f:
        history = json.load(f)
except:
    history = None

# Nombres de las clases
class_names = [
    "Normal", "COVID-19", "Neumonía", "Tuberculosis",
    "Asma", "Enfisema", "Fibrosis Pulmonar", "Bronquitis", "Otras"
]

def get_img_array(img, size=(224, 224)):
    img = img.resize(size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(array)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No se encontró ninguna capa Conv2D en el modelo.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        output = predictions[:, pred_index]

    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    return tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap), predictions[0].numpy()

# Sidebar: Datos del paciente
st.sidebar.header("Información del paciente")
nombre = st.sidebar.text_input("Nombre")
edad = st.sidebar.number_input("Edad", min_value=0, max_value=120, value=30)
sintomas = st.sidebar.text_area("Síntomas", height=100)
medico = st.sidebar.text_input("Médico responsable")

# Subir imagen
uploaded_file = st.file_uploader("Seleccioná una imagen de rayos X", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", width=300)

    try:
        # Procesamiento y Grad-CAM
        img_array = get_img_array(image)
        heatmap, predictions = make_gradcam_heatmap(img_array, model)

        # Superposición del heatmap
        img = np.array(image.resize((224, 224)))
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.image(superimposed_img, caption="Grad-CAM Superpuesto", width=300)

        # Mostrar información del paciente
        st.markdown("---")
        st.subheader("Datos del paciente")
        st.write(f"**Nombre:** {nombre if nombre else 'No especificado'}")
        st.write(f"**Edad:** {edad}")
        st.write(f"**Síntomas:** {sintomas if sintomas else 'No especificados'}")
        st.write(f"**Médico responsable:** {medico if medico else 'No especificado'}")

        # Predicción
        pred_class = int(np.argmax(predictions))
        pred_label = class_names[pred_class]
        confidence = float(np.max(predictions)) * 100
        st.markdown(f"### Clase predicha: {pred_label} ({confidence:.2f}%)")

        # Generar informe PDF
        if st.button("📄 Generar informe PDF"):
            pdf_path = "informe_paciente.pdf"
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image)
                ax.axis("off")
                ax.set_title("Imagen original")
                pdf.savefig(fig)
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                ax.axis("off")
                ax.set_title("Grad-CAM Superpuesto")
                pdf.savefig(fig)
                plt.close()

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis("off")
                text = f"Nombre: {nombre}\nEdad: {edad}\nSíntomas: {sintomas}\nClase predicha: {pred_label} ({confidence:.2f}%)\nMédico: {medico}"
                ax.text(0, 0.5, text, fontsize=12, verticalalignment='center')
                pdf.savefig(fig)
                plt.close()

            st.success("Informe PDF generado correctamente.")
            with open(pdf_path, "rb") as f:
                st.download_button("Descargar informe PDF", f, file_name=pdf_path, mime="application/pdf")

        # Mostrar historial si existe
        if history:
            st.subheader("Gráficas de Entrenamiento")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history['accuracy'], label='Entrenamiento')
            ax[0].plot(history['val_accuracy'], label='Validación')
            ax[0].set_title('Precisión')
            ax[0].legend()
            ax[1].plot(history['loss'], label='Entrenamiento')
            ax[1].plot(history['val_loss'], label='Validación')
            ax[1].set_title('Pérdida')
            ax[1].legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

else:
    st.info("Esperando una imagen de rayos X para iniciar el análisis.")