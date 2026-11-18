"""
app.py
------
Pokédex Clásica de Kanto (1ra Temporada) en Streamlit conectada a un modelo
de clasificación de imágenes Keras (.h5 / .keras) para 142 especies de Kanto.
"""

import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Pokédex Kanto | Clasificador Keras",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SPRITES_DIR = ASSETS_DIR / "sprites"
DATA_FILE = ASSETS_DIR / "pokemon_data.json"
BANNER_FILE = ASSETS_DIR / "kanto_banner.png"
TFLITE_FILE = BASE_DIR / "model.tflite"

# Lista exacta de las 142 clases del modelo en orden alfabético (índices 0 al 141)
POKEMON_CLASSES: List[str] = [
    "Abra", "Aerodactyl", "Alakazam", "Arbok", "Arcanine", "Articuno",
    "Beedrill", "Bellsprout", "Blastoise", "Bulbasaur", "Butterfree",
    "Caterpie", "Chansey", "Charizard", "Charmander", "Charmeleon",
    "Clefable", "Clefairy", "Cloyster", "Cubone", "Dewgong", "Diglett",
    "Ditto", "Dodrio", "Doduo", "Dragonair", "Dragonite", "Dratini",
    "Drowzee", "Dugtrio", "Eevee", "Ekans", "Electabuzz", "Electrode",
    "Exeggcute", "Exeggutor", "Farfetchd", "Fearow", "Flareon", "Gastly",
    "Gengar", "Geodude", "Gloom", "Golbat", "Goldeen", "Golduck",
    "Graveler", "Grimer", "Growlithe", "Gyarados", "Haunter", "Hitmonchan",
    "Hitmonlee", "Horsea", "Hypno", "Ivysaur", "Jigglypuff", "Jolteon",
    "Jynx", "Kabutops", "Kadabra", "Kakuna", "Kangaskhan", "Kingler",
    "Koffing", "Lapras", "Lickitung", "Machamp", "Machoke", "Machop",
    "Magikarp", "Magmar", "Magnemite", "Magneton", "Mankey", "Marowak",
    "Meowth", "Metapod", "Mew", "Mewtwo", "Moltres", "Mr. Mime",
    "Nidoking", "Nidoqueen", "Nidorina", "Nidorino", "Ninetales", "Oddish",
    "Omanyte", "Omastar", "Parasect", "Pidgeot", "Pidgeotto", "Pidgey",
    "Pikachu", "Pinsir", "Poliwag", "Poliwhirl", "Poliwrath", "Ponyta",
    "Porygon", "Primeape", "Psyduck", "Raichu", "Rapidash", "Raticate",
    "Rattata", "Rhydon", "Rhyhorn", "Sandshrew", "Sandslash", "Scyther",
    "Seadra", "Seaking", "Seel", "Shellder", "Slowbro", "Slowpoke",
    "Snorlax", "Spearow", "Squirtle", "Starmie", "Staryu", "Tangela",
    "Tauros", "Tentacool", "Tentacruel", "Vaporeon", "Venomoth", "Venonat",
    "Venusaur", "Victreebel", "Vileplume", "Voltorb", "Vulpix", "Wartortle",
    "Weedle", "Weepinbell", "Weezing", "Wigglytuff", "Zapdos", "Zubat",
]

# Colores oficiales por tipo de Pokémon para las píldoras
TYPE_COLORS: Dict[str, str] = {
    "Normal": "#A8A77A",
    "Fire": "#EE8130",
    "Water": "#6390F0",
    "Electric": "#F7D02C",
    "Grass": "#7AC74C",
    "Ice": "#96D9D6",
    "Fighting": "#C22E28",
    "Poison": "#A33EA1",
    "Ground": "#E2BF65",
    "Flying": "#A98FF3",
    "Psychic": "#F95587",
    "Bug": "#A6B91A",
    "Rock": "#B6A136",
    "Ghost": "#735797",
    "Dragon": "#6F35FC",
    "Dark": "#705746",
    "Steel": "#B7B7CE",
    "Fairy": "#D685AD",
}

STAT_ORDER: List[str] = [
    "HP",
    "Attack",
    "Defense",
    "Special Attack",
    "Special Defense",
    "Speed",
]


# ==============================================================================
# ESTILOS CSS POKÉDEX CLÁSICA (1RA TEMPORADA - KANTO)
# ==============================================================================
def inject_pokedex_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

        /* Fondo principal estilo chasis rojo clásico de Pokédex */
        .stApp {
            background: linear-gradient(160deg, #dc0a2d 0%, #b80724 55%, #8b051b 100%);
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
        }

        /* Contenedor principal con borde y sombra de consola retro */
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

        /* Ocultar decoración superior por defecto */
        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Barra superior de luces Pokédex */
        .pokedex-top-bar {
            display: flex;
            align-items: center;
            gap: 14px;
            background: linear-gradient(90deg, #99061e 0%, #c80a2a 50%, #99061e 100%);
            border: 3px solid #4a020e;
            border-radius: 16px;
            padding: 12px 20px;
            margin-bottom: 14px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35), inset 0 2px 4px rgba(255,255,255,0.2);
        }

        .pokedex-lens {
            width: 46px;
            height: 46px;
            border-radius: 50%;
            background: radial-gradient(circle at 32% 32%, #bae6fd 0%, #0ea5e9 45%, #0369a1 100%);
            border: 4px solid #f1f5f9;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.75), inset 0 0 6px rgba(0,0,0,0.4);
            flex-shrink: 0;
        }

        .pokedex-led {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid #2b0209;
            box-shadow: inset 0 2px 3px rgba(255,255,255,0.5);
        }
        .led-red { background: #ef4444; }
        .led-yellow { background: #facc15; }
        .led-green { background: #22c55e; }

        .pokedex-header-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 2px;
            color: #ffffff;
            text-shadow: 2px 2px 0px #4a020e;
            margin-left: 8px;
        }

        .pokedex-header-sub {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            color: #fef08a;
            margin-left: auto;
            letter-spacing: 1px;
        }

        /* Cabecera Kanto limpia con sprites laterales responsivos */
        .kanto-hero-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            background: linear-gradient(135deg, #b91c1c 0%, #991b1b 55%, #450a0a 100%);
            border: 3px solid #4a020e;
            border-radius: 16px;
            padding: 12px 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.14);
            position: relative;
            overflow: hidden;
        }

        .kanto-hero-sprites {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
            z-index: 2;
        }

        .kanto-hero-sprites img {
            width: 60px;
            height: 60px;
            object-fit: contain;
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.6));
            transition: transform 0.2s ease;
        }

        .kanto-hero-sprites img:hover {
            transform: scale(1.12);
        }

        .kanto-hero-center {
            text-align: center;
            flex: 1;
            z-index: 2;
            padding: 0 8px;
        }

        .kanto-hero-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: 2.5px;
            color: #ffffff;
            text-shadow: 0 3px 6px rgba(0, 0, 0, 0.85), 2px 2px 0px #450a0a;
            margin: 0 0 4px 0;
        }

        .kanto-hero-sub {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.92rem;
            color: #fef08a;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.85);
            letter-spacing: 1px;
            margin: 0;
        }

        @media (max-width: 900px) {
            .kanto-hero-sprites img {
                width: 42px;
                height: 42px;
            }
            .kanto-hero-title {
                font-size: 1.25rem;
            }
        }

        /* Títulos de columnas */
        .col-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #fef08a;
            background: rgba(20, 10, 15, 0.45);
            border-left: 4px solid #facc15;
            padding: 7px 12px;
            border-radius: 6px;
            margin-bottom: 12px;
        }

        /* File uploader estilizado */
        div[data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.55);
            border: 2px dashed #fca5a5;
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 14px;
        }
        div[data-testid="stFileUploader"] label {
            color: #fef2f2 !important;
            font-family: 'Share Tech Mono', monospace !important;
            font-size: 0.95rem !important;
        }

        /* ==============================================================
           COLUMNA 1: MARCO CLÁSICO DE LA PANTALLA POKÉDEX (BISEL GRIS/BLANCO)
           ============================================================== */
        .scanner-bezel {
            background: linear-gradient(145deg, #f1f5f9 0%, #cbd5e1 100%);
            border: 4px solid #334155;
            border-radius: 16px 16px 16px 46px;
            padding: 16px 20px 18px 20px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.45), inset 0 2px 4px rgba(255,255,255,0.9);
        }

        .bezel-top-leds {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 18px;
            margin-bottom: 12px;
        }

        .bezel-small-led {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, #fca5a5, #dc2626);
            border: 1.5px solid #450a0a;
            box-shadow: 0 0 6px rgba(239, 68, 68, 0.7);
        }

        .scanner-screen {
            background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
            border: 4px solid #1e293b;
            border-radius: 10px;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 12px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 0 18px rgba(0, 0, 0, 0.85);
        }

        .scanner-screen img {
            max-width: 100%;
            max-height: 255px;
            object-fit: contain;
            border-radius: 6px;
        }

        .scanner-placeholder {
            font-family: 'Share Tech Mono', monospace;
            color: #38bdf8;
            text-align: center;
            font-size: 0.95rem;
            line-height: 1.5;
            opacity: 0.85;
        }

        .scanner-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.75rem;
            color: #94a3b8;
            letter-spacing: 1px;
            margin-top: 8px;
            text-transform: uppercase;
        }

        .bezel-bottom-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 14px;
            padding: 0 8px;
        }

        .bezel-red-button {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, #f87171, #dc2626);
            border: 2.5px solid #450a0a;
            box-shadow: 0 3px 6px rgba(0,0,0,0.4);
        }

        .speaker-grille {
            display: flex;
            flex-direction: column;
            gap: 4px;
            width: 68px;
        }

        .speaker-slot {
            height: 4px;
            background: #334155;
            border-radius: 2px;
            box-shadow: inset 0 1px 1px rgba(0,0,0,0.6);
        }

        /* ==============================================================
           COLUMNA 2: IDENTIFICACIÓN Y AUDIO
           ============================================================== */
        .id-panel {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 4px solid #334155;
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
            margin-bottom: 12px;
        }

        .artwork-stage {
            background: radial-gradient(circle, rgba(56, 189, 248, 0.16) 0%, rgba(15, 23, 42, 0.9) 72%);
            border: 2px solid #1e293b;
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 14px;
        }

        .artwork-stage img {
            max-height: 235px;
            width: auto;
            object-fit: contain;
            filter: drop-shadow(0 10px 16px rgba(0,0,0,0.65));
        }

        .predicted-name {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.65rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: 1px;
            margin: 4px 0 8px 0;
        }

        .confidence-badge {
            display: inline-block;
            font-family: 'Share Tech Mono', monospace;
            font-size: 1rem;
            color: #052e16;
            background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
            padding: 5px 14px;
            border-radius: 999px;
            font-weight: 700;
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.45);
        }

        .cry-box-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            color: #fef08a;
            margin: 6px 0 4px 0;
            letter-spacing: 1px;
        }

        /* ==============================================================
           COLUMNA 3: FICHA TÉCNICA Y ESTADÍSTICAS
           ============================================================== */
        .dex-card {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border: 4px solid #334155;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
        }

        .dex-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #334155;
            padding-bottom: 12px;
            margin-bottom: 14px;
        }

        .dex-card-name {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0;
        }

        .dex-card-number {
            font-family: 'Share Tech Mono', monospace;
            font-size: 1.35rem;
            color: #facc15;
            background: rgba(250, 204, 21, 0.12);
            border: 1px solid rgba(250, 204, 21, 0.4);
            padding: 4px 12px;
            border-radius: 8px;
        }

        .dex-category-es {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.92rem;
            color: #38bdf8;
            letter-spacing: 0.8px;
            margin: -6px 0 12px 0;
        }

        .dex-description-box {
            background: rgba(15, 23, 42, 0.85);
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 14px;
            font-size: 0.92rem;
            line-height: 1.48;
            color: #e2e8f0;
        }

        .types-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }

        .type-pill {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #ffffff;
            padding: 5px 14px;
            border-radius: 999px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            box-shadow: 0 3px 8px rgba(0,0,0,0.35);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 16px;
        }

        .metric-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 10px 8px;
            text-align: center;
        }

        .metric-label {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .abilities-box {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 11px 14px;
            margin-bottom: 16px;
            font-size: 0.9rem;
        }

        .abilities-title {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.82rem;
            color: #facc15;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }

        .ability-tag {
            display: inline-block;
            background: #1e293b;
            border: 1px solid #475569;
            color: #e2e8f0;
            padding: 3px 10px;
            border-radius: 6px;
            margin-right: 6px;
            margin-bottom: 4px;
            font-size: 0.84rem;
        }

        .hidden-ability-tag {
            display: inline-block;
            background: rgba(168, 85, 247, 0.18);
            border: 1px solid #a855f7;
            color: #e9d5ff;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.84rem;
        }

        /* Barras de estadísticas base con degradado rojo -> naranja -> amarillo -> verde */
        .stats-title {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            color: #facc15;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .stat-row {
            display: grid;
            grid-template-columns: 125px 42px 1fr;
            align-items: center;
            gap: 10px;
            margin-bottom: 9px;
        }

        .stat-name {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.84rem;
            color: #cbd5e1;
        }

        .stat-val {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            color: #ffffff;
            text-align: right;
        }

        .stat-bar-bg {
            width: 100%;
            height: 12px;
            background: #090d16;
            border: 1px solid #334155;
            border-radius: 999px;
            overflow: hidden;
        }

        .stat-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #ef4444 0%, #f97316 33%, #eab308 66%, #22c55e 100%);
            box-shadow: 0 0 8px rgba(250, 204, 21, 0.35);
            transition: width 0.4s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# CARGA DE MODELO Y DATOS
# ==============================================================================
def find_model_file() -> Optional[Path]:
    """Busca automáticamente el primer archivo .keras o .h5 en el directorio actual."""
    keras_files = sorted(BASE_DIR.glob("*.keras"))
    h5_files = sorted(BASE_DIR.glob("*.h5"))
    candidates = keras_files + h5_files
    return candidates[0] if candidates else None


@st.cache_resource(show_spinner="Cargando motor neuronal TFLite de Kanto...")
def load_inference_engine() -> dict:
    """
    Carga el motor de inferencia UNA sola vez globalmente (sin argumentos variables).
    Convierte automáticamente el modelo Keras a TFLite float32 (`model.tflite`) la primera vez
    para lograr inferencia instantánea (<0.3s) en CPU y ejecuta warm-up inicial.
    """
    import tensorflow as tf

    t_start = time.perf_counter()

    if not TFLITE_FILE.exists():
        model_path = find_model_file()
        if model_path is None:
            raise FileNotFoundError("No se encontró ningún archivo `.keras` o `.h5` en la carpeta raíz.")
        keras_model = tf.keras.models.load_model(str(model_path), compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_bytes = converter.convert()
        TFLITE_FILE.write_bytes(tflite_bytes)
        print(f"[TIMING] Conversión Keras -> TFLite completada en {time.perf_counter() - t_start:.3f}s")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_FILE), num_threads=4)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Warm-up con ceros para compilar/inicializar buffers al arranque
    interpreter.set_tensor(input_index, np.zeros((1, 224, 224, 3), dtype=np.float32))
    interpreter.invoke()

    print(f"[TIMING] Carga + warm-up TFLite: {time.perf_counter() - t_start:.3f}s")
    return {
        "backend": "tflite",
        "interpreter": interpreter,
        "input_index": input_index,
        "output_index": output_index,
    }


@st.cache_data(show_spinner=False)
def load_pokemon_data() -> Dict[str, dict]:
    """Carga el diccionario de metadatos desde assets/pokemon_data.json (sin llamadas HTTP)."""
    if not DATA_FILE.exists():
        return {}
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_sprite_b64(slug: str) -> str:
    """Carga un sprite local desde assets/sprites/ en base64 para decoración del banner."""
    path = SPRITES_DIR / f"{slug}.png"
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ==============================================================================
# PREPROCESAMIENTO E INFERENCIA
# ==============================================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen subida por el usuario para el modelo (224x224 RGB float32).
    1. Convierte obligatoriamente a RGB (elimina canal alfa RGBA en PNGs).
    2. Redimensiona a 224x224 píxeles.
    3. Aplica el rescaling / centrado ImageNet (Caffe BGR) esperado por ResNet50.
    """
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.asarray(resized, dtype=np.float32)

    # RESCALING / NORMALIZACIÓN (equivalente exacto a resnet50.preprocess_input en modo Caffe):
    # Convierte RGB -> BGR y resta las medias de ImageNet [103.939, 116.779, 123.68] sin dividir por 255.
    # Si tu modelo fuera entrenado en [0, 1], reemplaza por: img_array = img_array / 255.0
    bgr_array = img_array[..., ::-1].copy()
    bgr_array[..., 0] -= 103.939
    bgr_array[..., 1] -= 116.779
    bgr_array[..., 2] -= 123.68
    return np.expand_dims(bgr_array, axis=0)


def predict_pokemon(engine: dict, image: Image.Image) -> Tuple[str, float, List[Tuple[str, float]], dict]:
    """Ejecuta el preprocesamiento y la inferencia TFLite (<0.3s) midiendo tiempos en consola."""
    t0 = time.perf_counter()
    batch = preprocess_image(image)
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    interpreter = engine["interpreter"]
    interpreter.set_tensor(engine["input_index"], batch)
    interpreter.invoke()
    preds = interpreter.get_tensor(engine["output_index"])[0]
    t_infer = time.perf_counter() - t1

    # Si la última capa devuelve logits en lugar de probabilidades softmax, normalizamos
    if np.min(preds) < 0 or not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / np.sum(exp_preds)
    else:
        probs = preds

    top_indices = np.argsort(probs)[::-1][:3]
    best_idx = int(top_indices[0])
    best_name = POKEMON_CLASSES[best_idx]
    best_conf = float(probs[best_idx]) * 100.0
    top_3 = [(POKEMON_CLASSES[int(i)], float(probs[int(i)]) * 100.0) for i in top_indices]
    timings = {"preprocess_ms": t_prep * 1000.0, "infer_ms": t_infer * 1000.0}
    return best_name, best_conf, top_3, timings


def image_to_base64(image: Image.Image) -> str:
    """Codifica una imagen PIL a base64 PNG para incrustarla dentro de la pantalla CSS."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================
def main() -> None:
    inject_pokedex_css()

    # Barra superior clásica de la Pokédex
    st.markdown(
        """
        <div class="pokedex-top-bar">
            <div class="pokedex-lens"></div>
            <div class="pokedex-led led-red"></div>
            <div class="pokedex-led led-yellow"></div>
            <div class="pokedex-led led-green"></div>
            <div class="pokedex-header-title">POKÉDEX KANTO • GEN I</div>
            <div class="pokedex-header-sub">OS v1.0 // 142 ESPECIES</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cabecera Hero de Kanto con sprites icónicos laterales (100% locales en base64)
    left_icons = ["bulbasaur", "charmander", "squirtle"]
    right_icons = ["pikachu", "eevee", "charizard"]
    left_imgs_html = "".join(
        f'<img src="data:image/png;base64,{load_sprite_b64(s)}" alt="{s}" />'
        for s in left_icons if load_sprite_b64(s)
    )
    right_imgs_html = "".join(
        f'<img src="data:image/png;base64,{load_sprite_b64(s)}" alt="{s}" />'
        for s in right_icons if load_sprite_b64(s)
    )

    st.markdown(
        f"""
        <div class="kanto-hero-banner">
            <div class="kanto-hero-sprites">{left_imgs_html}</div>
            <div class="kanto-hero-center">
                <div class="kanto-hero-title">POKÉDEX KANTO</div>
                <div class="kanto-hero-sub">PRIMERA TEMPORADA • SISTEMA DE IDENTIFICACIÓN VISUAL (142 ESPECIES)</div>
            </div>
            <div class="kanto-hero-sprites">{right_imgs_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Verificar datos locales y cargar motor TFLite cacheado
    t_assets_start = time.perf_counter()
    pokemon_data = load_pokemon_data()
    t_json_ms = (time.perf_counter() - t_assets_start) * 1000.0
    if not pokemon_data:
        st.warning(
            "No se encontró `assets/pokemon_data.json`. Ejecuta primero `python setup_data.py` "
            "para descargar las estadísticas, sprites y sonidos de PokéAPI."
        )

    t_model_start = time.perf_counter()
    try:
        engine = load_inference_engine()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    t_model_ms = (time.perf_counter() - t_model_start) * 1000.0

    # Estructura en 3 Columnas principales
    col_scanner, col_ident, col_stats = st.columns([1.05, 1.0, 1.35], gap="large")

    uploaded_image: Optional[Image.Image] = None
    predicted_name: Optional[str] = None
    confidence: float = 0.0
    top_3: List[Tuple[str, float]] = []
    timings: Dict[str, float] = {}

    # --------------------------------------------------------------------------
    # COLUMNA 1: ESCÁNER - PANTALLA DE LA POKÉDEX
    # --------------------------------------------------------------------------
    with col_scanner:
        st.markdown('<div class="col-title">1. Escáner Óptico</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Sube una imagen de Pokémon (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="pokemon_uploader",
        )

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            img_b64 = image_to_base64(uploaded_image)
            screen_content = (
                f'<img src="data:image/png;base64,{img_b64}" alt="Scanned image" />'
                '<div class="scanner-label">Scanned image • RGB 224x224</div>'
            )
            predicted_name, confidence, top_3, timings = predict_pokemon(engine, uploaded_image)
        else:
            screen_content = (
                '<div class="scanner-placeholder">'
                'SISTEMA EN ESPERA...<br><br>'
                'Carga una imagen arriba para<br>iniciar el escaneo de Kanto.'
                '</div>'
            )

        st.markdown(
            f"""
            <div class="scanner-bezel">
                <div class="bezel-top-leds">
                    <div class="bezel-small-led"></div>
                    <div class="bezel-small-led"></div>
                </div>
                <div class="scanner-screen">
                    {screen_content}
                </div>
                <div class="bezel-bottom-controls">
                    <div class="bezel-red-button"></div>
                    <div class="speaker-grille">
                        <div class="speaker-slot"></div>
                        <div class="speaker-slot"></div>
                        <div class="speaker-slot"></div>
                        <div class="speaker-slot"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # COLUMNA 2: IDENTIFICACIÓN Y AUDIO
    # --------------------------------------------------------------------------
    with col_ident:
        st.markdown('<div class="col-title">2. Identificación y Grito</div>', unsafe_allow_html=True)

        if predicted_name is not None:
            t_media_start = time.perf_counter()
            entry = pokemon_data.get(predicted_name, {})
            sprite_rel = entry.get("sprite", "")
            sprite_path = BASE_DIR / sprite_rel if sprite_rel else None

            if sprite_path and sprite_path.exists():
                artwork_b64 = base64.b64encode(sprite_path.read_bytes()).decode("utf-8")
                artwork_html = f'<img src="data:image/png;base64,{artwork_b64}" alt="{predicted_name}" />'
            else:
                artwork_html = '<div class="scanner-placeholder">Artwork no disponible</div>'

            st.markdown(
                f"""
                <div class="id-panel">
                    <div class="artwork-stage">
                        {artwork_html}
                    </div>
                    <div class="predicted-name">{predicted_name}</div>
                    <div class="confidence-badge">Confianza: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Reproductor automático de la Voz de la Pokédex (ES) estilo Dexter (.mp3)
            voice_rel = entry.get("voice_path", "")
            voice_path = BASE_DIR / voice_rel if voice_rel else None
            st.markdown('<div class="cry-box-label">🎙️ VOZ DE LA POKÉDEX (ES)</div>', unsafe_allow_html=True)
            if voice_path and voice_path.exists():
                with open(voice_path, "rb") as voice_file:
                    voice_bytes = voice_file.read()
                st.audio(voice_bytes, format="audio/mp3", autoplay=True, width="stretch")
            else:
                st.info("Voz (.mp3) no encontrada. Ejecuta `setup_data.py`.")

            # Reproductor separado del Grito / Cry original del videojuego (.ogg)
            audio_rel = entry.get("audio_path") or entry.get("cry", "")
            audio_path = BASE_DIR / audio_rel if audio_rel else None
            st.markdown('<div class="cry-box-label">🔊 GRITO / CRY ORIGINAL</div>', unsafe_allow_html=True)
            if audio_path and audio_path.exists():
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/ogg", width="stretch")
            else:
                st.info("Audio (.ogg) no encontrado. Ejecuta `setup_data.py`.")

            t_media_ms = (time.perf_counter() - t_media_start) * 1000.0
            print(
                f"[TIMING] {predicted_name} ({confidence:.2f}%) | "
                f"Modelo(cache): {t_model_ms:.1f}ms | "
                f"Preproc: {timings.get('preprocess_ms', 0):.1f}ms | "
                f"Inferencia TFLite: {timings.get('infer_ms', 0):.1f}ms | "
                f"Assets locales: {t_json_ms + t_media_ms:.1f}ms"
            )

            # Top 3 predicciones
            if top_3:
                with st.expander("Ver Top 3 predicciones del modelo"):
                    for rank, (cls_name, prob) in enumerate(top_3, start=1):
                        st.write(f"**#{rank} {cls_name}** — `{prob:.2f}%`")
        else:
            st.markdown(
                """
                <div class="id-panel" style="min-height: 360px; display: flex; flex-direction: column; justify-content: center;">
                    <div class="scanner-placeholder">
                        ESPERANDO IDENTIFICACIÓN...<br><br>
                        Aquí aparecerá el Artwork Oficial,<br>
                        el porcentaje de confianza y el grito.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # COLUMNA 3: FICHA TÉCNICA Y ESTADÍSTICAS
    # --------------------------------------------------------------------------
    with col_stats:
        st.markdown('<div class="col-title">3. Ficha Técnica y Estadísticas</div>', unsafe_allow_html=True)

        if predicted_name is not None and predicted_name in pokemon_data:
            info = pokemon_data[predicted_name]
            dex_num = info.get("id", 0)
            categoria_es = info.get("categoria_es", "Pokémon de Kanto")
            descripcion_es = info.get("descripcion_es", "")
            types = info.get("types", [])
            height_m = info.get("height_m", 0.0)
            weight_kg = info.get("weight_kg", 0.0)
            base_xp = info.get("base_experience", 0)
            abilities = info.get("abilities", [])
            hidden_ab = info.get("hidden_ability")
            stats = info.get("stats", {})

            type_pills_html = "".join(
                f'<span class="type-pill" style="background-color: {TYPE_COLORS.get(t, "#64748b")};">{t}</span>'
                for t in types
            )

            abilities_html = "".join(f'<span class="ability-tag">{ab}</span>' for ab in abilities)
            hidden_html = (
                f'<span class="hidden-ability-tag">★ Hidden: {hidden_ab}</span>'
                if hidden_ab
                else '<span class="ability-tag" style="opacity:0.6;">Sin habilidad oculta</span>'
            )

            stats_rows_html = ""
            for stat_label in STAT_ORDER:
                val = int(stats.get(stat_label, 0))
                # Escala visual sobre 180 (cubre el rango típico de stats base en Gen I)
                pct = min(100.0, (val / 180.0) * 100.0)
                stats_rows_html += (
                    f'<div class="stat-row">'
                    f'  <div class="stat-name">{stat_label}</div>'
                    f'  <div class="stat-val">{val}</div>'
                    f'  <div class="stat-bar-bg">'
                    f'    <div class="stat-bar-fill" style="width: {pct:.1f}%;"></div>'
                    f'  </div>'
                    f'</div>'
                )

            st.markdown(
                f"""
                <div class="dex-card">
                    <div class="dex-card-header">
                        <h2 class="dex-card-name">{predicted_name}</h2>
                        <span class="dex-card-number">No. {dex_num:03d}</span>
                    </div>
                    <div class="dex-category-es">{categoria_es}</div>
                    <div class="types-row">{type_pills_html}</div>
                    <div class="dex-description-box">{descripcion_es}</div>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-label">Height</div>
                            <div class="metric-value">{height_m} m</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Weight</div>
                            <div class="metric-value">{weight_kg} kg</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Base XP</div>
                            <div class="metric-value">{base_xp}</div>
                        </div>
                    </div>
                    <div class="abilities-box">
                        <div class="abilities-title">Abilities & Hidden Ability</div>
                        <div>{abilities_html} {hidden_html}</div>
                    </div>
                    <div class="stats-title">Base Stats</div>
                    {stats_rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="dex-card" style="min-height: 360px; display: flex; flex-direction: column; justify-content: center;">
                    <div class="scanner-placeholder">
                        FICHA TÉCNICA INACTIVA<br><br>
                        Escanea un Pokémon para consultar<br>
                        su número nacional, tipos, medidas,<br>
                        habilidades y estadísticas base.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
