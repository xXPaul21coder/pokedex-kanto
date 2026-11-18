# ⚡ Pokédex Kanto AI: Clasificador de Pokémon Gen-1 con ResNet50 & Streamlit

Aplicación web interactiva de Visión por Computadora capaz de identificar **142 especies de Pokémon de la región de Kanto (Primera Generación)** en tiempo real. El sistema está impulsado por una arquitectura **ResNet50** entrenada mediante *Transfer Learning* y *Fine-Tuning* en TensorFlow/Keras, optimizada a **TensorFlow Lite (`model.tflite`)** para inferencia ligera en producción.

🚀 **[Probar la Web App en vivo (Streamlit Cloud)](https://pokedex-kanto.streamlit.app/)**

---

## 🎯 Sobre el Proyecto
Reconocer Pokémon a partir de imágenes presenta un desafío de **alta granularidad visual**: dentro de una misma especie coexisten ilustraciones oficiales, *sprites* retro, fotogramas de anime, cartas coleccionables y modelos 3D (alta variabilidad intra-clase), mientras que especies pertenecientes a una misma línea evolutiva comparten paleta cromática y rasgos anatómicos (alta similitud inter-clase).

El flujo completo del proyecto abarca:
1. **Análisis Exploratorio (EDA) y Depuración:** Auditoría de integridad de archivos, canales de color y consolidación de categorías duplicadas (17,261 imágenes válidas en 142 clases).
2. **Pipeline de Datos (`tf.data`):** Partición estratificada (70/15/15), conversión uniforme a `224×224×3` RGB, normalización estándar de ResNet50 y *Data Augmentation* en entrenamiento.
3. **Entrenamiento en 2 Etapas y Experimentación:** Extracción de características con backbone congelado + *Fine-Tuning* selectivo del bloque residual `conv5_x`, comparando optimizadores (`Adam` vs. `AdamW`) y tasas de `Dropout`.
4. **Optimización y Despliegue:** Exportación del modelo ganador a **TensorFlow Lite (`model.tflite`)** e integración en una interfaz web con **Streamlit**.

---

## 📂 Estructura del Repositorio

```text
pokedex-kanto/
├── assets/             # Recursos visuales e imágenes de interfaz para la app
├── app.py              # Aplicación principal en Streamlit e inferencia TFLite
├── model.tflite        # Modelo ResNet50 ganador exportado y optimizado para producción
└── requirements.txt    # Dependencias del entorno para ejecución y despliegue
```
*(Nota: Los notebooks de Análisis Exploratorio, entrenamiento experimental y evaluación detallada se desarrollaron en Google Colab).*

---

## 📊 Dataset y Preprocesamiento
* **Fuente:** Kaggle (`pokemon-images-first-generation17000-files`) — **17,261 imágenes válidas** (0 archivos corruptos).
* **Limpieza de etiquetas:** Unificación de carpetas duplicadas (`Mr. Mime` y `MrMime`), consolidando **142 especies únicas**.
* **Partición estratificada (`SEED = 42`):**
  * **Entrenamiento (70%):** `12,082` imágenes
  * **Validación (15%):** `2,589` imágenes
  * **Prueba / Test (15%):** `2,590` imágenes
* **Tubería `tf.data`:** Decodificación forzada a 3 canales RGB (`channels=3`) para estandarizar imágenes `RGBA`, paleta indexada `P` y escala de grises `L/LA`, redimensionamiento a `224×224` px y aumento de datos (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`) aplicado exclusivamente en entrenamiento.

---

## 🧠 Arquitectura y Experimentación
Se empleó **ResNet50** preentrenada en ImageNet (`include_top=False`) acoplada a un cabezal clasificador regularizado: `GlobalAveragePooling2D` → `Dense(512, ReLU)` → `BatchNormalization` → `Dropout` → `Dense(142, Softmax)`.

**Protocolo de entrenamiento en 2 etapas:**
1. **Etapa 1 (Feature Extraction):** Backbone congelado (`lr=1e-3`) con `EarlyStopping` y `ReduceLROnPlateau`.
2. **Etapa 2 (Fine-Tuning):** Descongelamiento selectivo del bloque superior `conv5_x` (37 capas entrenables) con `lr=1e-5` para adaptar las representaciones semánticas al dominio.

### Comparativa de Hiperparámetros (Validation Set)
| Configuración | Optimizador | Dropout | Val Loss | Val Top-1 Acc | Val Top-5 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Config_Base | Adam | 0.4 | 0.4069 | 90.23% | 97.88% |
| **Config_A (Modelo en Producción)** | **AdamW** (`wd=1e-4`) | **0.3** | **0.3485** | **91.46%** | **98.03%** |
| Config_B | AdamW (`wd=1e-4`) | 0.5 | 0.3414 | 91.08% | 98.22% |

---

## 📈 Resultados en Test (2,590 imágenes inéditas)
* **Top-1 Accuracy:** `90.04%`
* **Top-5 Accuracy:** `97.49%`
* **Test Loss:** `0.3901`

### 🔍 Hallazgos del Análisis de Confusión
* **Independencia cromática:** A diferencia de descriptores basados en histogramas RGB (incapaces de separar especies rosadas como *Jigglypuff*, *Clefairy* y *Chansey*), la red aprendió contornos, extremidades y proporciones espaciales.
* **Líneas evolutivas:** Los errores residuales se concentran entre etapas consecutivas de una misma familia evolutiva que comparten anatomía y color (ej. *Nidoran* / *Nidorina*, *Gastly* / *Haunter*, *Poliwag* / *Poliwhirl*), casos que el **97.49% de Top-5 Accuracy** captura casi en su totalidad.

---

## 🛠️ Stack Tecnológico
* **Deep Learning & Optimización:** TensorFlow 2.21, Keras, TensorFlow Lite (`TFLite`)
* **Procesamiento & Ciencia de Datos:** NumPy, Pandas, Scikit-Learn, Pillow
* **Frontend & Despliegue:** Streamlit, Streamlit Cloud

---

## 🚀 Instalación y Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/xXPaul21coder/pokedex-kanto.git
   cd pokedex-kanto
   ```

2. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py
   ```

---

## 👨‍💻 Autor
**Paul Cari**  
Ingeniero Industrial especializado en Ciencia de Datos | Estudiante de Maestría en Inteligencia Artificial  
* [![GitHub](https://img.shields.io/badge/GitHub-xXPaul21coder-181717?style=flat&logo=github)](https://github.com/xXPaul21coder)
* [![App](https://img.shields.io/badge/Streamlit-Pokédex%20Kanto%20Live-FF4B4B?style=flat&logo=streamlit)](https://pokedex-kanto.streamlit.app/)