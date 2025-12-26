#!/usr/bin/env python3
"""
Application Web Streamlit - Segmentation Cityscapes
Future Vision Transport - Démo de l'API de Segmentation

Cette application permet de tester l'API de segmentation sémantique
via une interface web interactive.

Usage:
    streamlit run app_streamlit.py --server.port 8501
"""

import streamlit as st
import requests
import base64
import io
import json
import os
from glob import glob
from PIL import Image
import numpy as np

# Configure matplotlib pour mode headless (sans display)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# URL de l'API (à modifier selon votre déploiement)
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Palette de couleurs - 8 CATÉGORIES (basée sur config.py du projet)
COLOR_PALETTE = np.array([
    [0, 0, 0],           # 0: void - noir
    [128, 64, 128],      # 1: flat - violet
    [70, 70, 70],        # 2: construction - gris
    [153, 153, 153],     # 3: object - gris clair
    [107, 142, 35],      # 4: nature - vert olive
    [70, 130, 180],      # 5: sky - bleu ciel
    [220, 20, 60],       # 6: human - rouge
    [0, 0, 142],         # 7: vehicle - bleu foncé
], dtype=np.uint8)

# Noms des 8 CATÉGORIES (utilisées par le modèle)
CLASS_NAMES = [
    'void', 'flat', 'construction', 'object',
    'nature', 'sky', 'human', 'vehicle'
]

# Mapping des IDs Cityscapes originaux (0-33) vers les 8 catégories
ID_TO_CATEGORY = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,  # void
    7: 1, 8: 1, 9: 1, 10: 1,                    # flat
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,  # construction
    17: 3, 18: 3, 19: 3, 20: 3,                 # object
    21: 4, 22: 4,                               # nature
    23: 5,                                       # sky
    24: 6, 25: 6,                               # human
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,  # vehicle
}

# ============================================================================
# Configuration de la Page
# ============================================================================

st.set_page_config(
    page_title="Cityscapes Segmentation Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec dark mode
def get_custom_css(dark_mode=False):
    if dark_mode:
        return """
<style>
    /* Dark Mode - GitHub Style */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* Streamlit Header (bandeau en haut) */
    header[data-testid="stHeader"] {
        background-color: #161b22 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #c9d1d9;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Expanders (À propos) */
    [data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    [data-testid="stExpander"] summary {
        color: #c9d1d9 !important;
    }
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] div {
        color: #c9d1d9 !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #30363d;
    }

    /* Success/Warning Boxes */
    .success-box {
        background-color: #0d1117;
        border-left: 5px solid #238636;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #7ee787;
    }
    .warning-box {
        background-color: #0d1117;
        border-left: 5px solid #d29922;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #e3b341;
    }

    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white !important;
        border: 1px solid #2ea043;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
    }
    .stTabs [data-baseweb="tab"] {
        color: #c9d1d9;
    }

    /* Text elements */
    p, span, label, div {
        color: #c9d1d9;
    }

    /* Select boxes - FIXED CONTRAST */
    .stSelectbox label {
        color: #c9d1d9 !important;
    }
    .stSelectbox > div > div {
        background-color: #161b22 !important;
    }
    .stSelectbox input {
        color: #ffffff !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #ffffff !important;
    }

    /* Number input - FIXED CONTRAST */
    .stNumberInput label {
        color: #c9d1d9 !important;
    }
    .stNumberInput input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
    }

    /* All inputs */
    input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
    }
</style>
"""
    else:
        return """
<style>
    /* Light Mode */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
"""

# ============================================================================
# Fonctions Utilitaires
# ============================================================================

@st.cache_data
def check_api_health():
    """Vérifie l'état de santé de l'API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except Exception as e:
        return False, str(e)


@st.cache_data
def get_api_classes():
    """Récupère les informations sur les classes de segmentation."""
    try:
        response = requests.get(f"{API_URL}/classes", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des classes : {e}")
        return None


def decode_base64_to_array(base64_str):
    """
    Décode une chaîne base64 en array numpy.

    Args:
        base64_str: Chaîne base64

    Returns:
        np.ndarray: Array numpy
    """
    import json
    decoded = base64.b64decode(base64_str)
    array_list = json.loads(decoded.decode('utf-8'))
    return np.array(array_list)


def decode_base64_to_image(base64_str):
    """
    Décode une chaîne base64 en image PIL.

    Args:
        base64_str: Chaîne base64

    Returns:
        PIL.Image: Image PIL
    """
    decoded = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(decoded))


def predict_segmentation(image_file, return_colored=True, return_raw_mask=False):
    """
    Envoie une image à l'API pour prédiction.

    Args:
        image_file: Fichier image uploadé
        return_colored: Si True, retourne le masque colorisé
        return_raw_mask: Si True, retourne le masque brut

    Returns:
        dict: Résultats de la prédiction
    """
    try:
        # Préparer le fichier avec le bon format
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        params = {
            "return_colored": return_colored,
            "return_raw_mask": return_raw_mask
        }

        # DEBUG: Afficher les informations de la requête
        st.info(f"🔍 DEBUG API - Envoi de la requête:\n"
                f"  URL: {API_URL}/predict\n"
                f"  Fichier: {image_file.name}\n"
                f"  Type: {image_file.type}\n"
                f"  Taille: {len(image_file.getvalue())} bytes\n"
                f"  return_colored: {return_colored}\n"
                f"  return_raw_mask: {return_raw_mask}")

        # Appel API
        with st.spinner("Segmentation en cours..."):
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                params=params,
                timeout=60
            )

        # DEBUG: Afficher les informations de la réponse
        st.info(f"🔍 DEBUG API - Réponse reçue:\n"
                f"  Status Code: {response.status_code}\n"
                f"  Headers: {dict(response.headers)}\n"
                f"  Response Length: {len(response.content)} bytes")

        if response.status_code == 200:
            try:
                json_data = response.json()
                st.success(f"✅ Réponse JSON valide - Clés: {list(json_data.keys())}")
                return True, json_data
            except Exception as json_err:
                st.error(f"❌ Erreur parsing JSON: {str(json_err)}\n"
                         f"Content: {response.text[:500]}")
                return False, f"Erreur parsing JSON: {str(json_err)}"
        else:
            error_msg = f"Erreur API : {response.status_code} - {response.text}"
            st.error(f"❌ {error_msg}")
            # Essayer de parser le texte d'erreur comme JSON pour plus de détails
            try:
                error_detail = response.json()
                st.code(f"Détails de l'erreur:\n{error_detail}", language="json")
            except:
                st.code(f"Réponse brute:\n{response.text}", language="text")
            return False, error_msg

    except Exception as e:
        error_msg = f"Erreur : {str(e)}"
        st.error(f"❌ Exception: {error_msg}")
        import traceback
        st.code(traceback.format_exc(), language="text")
        return False, error_msg


def get_prediction_image(image_file, overlay=False):
    """
    Récupère directement l'image de prédiction depuis l'API.

    Args:
        image_file: Fichier image uploadé
        overlay: Si True, retourne un overlay

    Returns:
        PIL.Image ou None
    """
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        params = {"overlay": overlay}

        response = requests.post(
            f"{API_URL}/predict/image",
            files=files,
            params=params,
            timeout=60
        )

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            st.error(f"Erreur API : {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Erreur : {str(e)}")
        return None


def colorize_mask(mask, is_cityscapes_gt=False):
    """
    Colorise un masque de segmentation avec la palette de 8 catégories.

    Args:
        mask: PIL Image ou numpy array avec des labels
        is_cityscapes_gt: Si True, applique le mapping IDs Cityscapes (0-33) → catégories (0-7)

    Returns:
        PIL Image colorisée
    """
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
    else:
        mask_array = mask.copy()

    # Détection automatique : si les valeurs max > 7, c'est un GT Cityscapes avec IDs originaux
    if not is_cityscapes_gt and mask_array.max() > 7:
        is_cityscapes_gt = True

    # Si c'est un GT Cityscapes, mapper les IDs originaux (0-33) vers catégories (0-7)
    if is_cityscapes_gt:
        mapped_mask = np.zeros(mask_array.shape, dtype=np.uint8)
        for original_id, category_id in ID_TO_CATEGORY.items():
            mapped_mask[mask_array == original_id] = category_id
        mask_array = mapped_mask

    # Créer une image RGB vide
    h, w = mask_array.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Appliquer la palette de couleurs pour les 8 catégories
    for category_id in range(len(COLOR_PALETTE)):
        colored[mask_array == category_id] = COLOR_PALETTE[category_id]

    return Image.fromarray(colored)


def find_ground_truth(image_name):
    """
    Cherche automatiquement le ground truth correspondant à une image.

    Supporte plusieurs formats :
    1. Cityscapes: [city]_[seq]_[frame]_leftImg8bit.png → [base]_gtFine_labelIds.png
    2. Augmented: [name]_image.png → [name]_label.png
    3. Augmented Cityscapes: [city]_[seq]_[frame]_aug.png → [base]_gtFine_labelIds.png

    Args:
        image_name: Nom de l'image

    Returns:
        PIL.Image ou None si non trouvé
    """
    import os
    from glob import glob

    # Chemins de recherche prioritaires
    search_paths = [
        "sample_data/labels",  # Sample data pour Streamlit Cloud
        "/home/ser/Bureau/Projet_image_new/P8_Cityscapes_gtFine_trainvaltest/gtFine",  # Dataset officiel local
        "/home/ser/Bureau/Projet_image_new/data",  # Dataset augmenté/personnalisé local
    ]
    gt_filename = None

    # Format 1: Images Cityscapes standard
    if "_leftImg8bit" in image_name:
        base_name = image_name.replace("_leftImg8bit.png", "").replace("_leftImg8bit.jpg", "").replace("_leftImg8bit.jpeg", "")
        gt_filename = f"{base_name}_gtFine_labelIds.png"

    # Format 2: Images augmentées (test_image.png → test_label.png)
    elif "_image." in image_name:
        base_name = image_name.replace("_image.png", "").replace("_image.jpg", "").replace("_image.jpeg", "")
        gt_filename = f"{base_name}_label.png"

    # Format 3: Images Cityscapes augmentées (cologne_000025_000019_aug.png → cologne_000025_000019_gtFine_labelIds.png)
    elif "_aug." in image_name or image_name.endswith("_aug.png"):
        base_name = image_name.replace("_aug.png", "").replace("_aug.jpg", "").replace("_aug.jpeg", "")
        gt_filename = f"{base_name}_gtFine_labelIds.png"

    # Format 4: Fichiers simples (test.png → test_label.png ou test_gt.png)
    else:
        # Essayer plusieurs suffixes possibles
        base_name = image_name.rsplit('.', 1)[0]  # Enlever l'extension
        possible_suffixes = ["_label", "_gt", "_gtFine_labelIds", "_mask"]

        for suffix in possible_suffixes:
            gt_filename = f"{base_name}{suffix}.png"

            # Chercher dans tous les chemins de recherche
            for search_path in search_paths:
                if not os.path.exists(search_path):
                    continue

                gt_pattern = f"{search_path}/**/{gt_filename}"
                matches = glob(gt_pattern, recursive=True)

                if matches:
                    try:
                        return Image.open(matches[0]).convert('L')
                    except:
                        pass

        return None

    # Chercher le fichier GT dans tous les chemins
    if gt_filename:
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue

            # Recherche récursive du fichier GT
            gt_pattern = f"{search_path}/**/{gt_filename}"
            matches = glob(gt_pattern, recursive=True)

            if matches:
                try:
                    # Charger et convertir en niveaux de gris pour les labels
                    img = Image.open(matches[0])
                    # Si l'image a une palette, la convertir d'abord
                    if img.mode == 'P':
                        img = img.convert('L')
                    elif img.mode != 'L':
                        img = img.convert('L')
                    return img
                except Exception as e:
                    print(f"Erreur lors du chargement du GT: {e}")
                    continue

    return None


def calculate_iou_metrics(pred_mask, gt_mask, num_classes=8):
    """
    Calcule les métriques IoU entre la prédiction et le ground truth.

    Args:
        pred_mask: Masque prédit (image PIL ou numpy array)
        gt_mask: Masque ground truth (image PIL ou numpy array)
        num_classes: Nombre de classes

    Returns:
        dict: Métriques IoU par classe et mIoU
    """
    # Convertir en numpy si nécessaire
    if isinstance(pred_mask, Image.Image):
        pred_mask = np.array(pred_mask)
    if isinstance(gt_mask, Image.Image):
        gt_mask = np.array(gt_mask)

    # Redimensionner le ground truth si nécessaire
    if pred_mask.shape != gt_mask.shape:
        gt_mask = np.array(Image.fromarray(gt_mask).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST
        ))

    # IMPORTANT: Convertir le GT des IDs Cityscapes (0-33) vers catégories (0-7)
    # Si le GT contient des valeurs > 7, c'est qu'il utilise les IDs originaux
    if gt_mask.max() > 7:
        mapped_gt = np.zeros(gt_mask.shape, dtype=np.uint8)
        for original_id, category_id in ID_TO_CATEGORY.items():
            mapped_gt[gt_mask == original_id] = category_id
        gt_mask = mapped_gt

    iou_per_class = {}
    ious = []

    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id)
        gt_class = (gt_mask == class_id)

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        if union > 0:
            iou = (intersection / union) * 100
            iou_per_class[class_id] = iou
            ious.append(iou)
        else:
            iou_per_class[class_id] = 0.0

    miou = np.mean(ious) if ious else 0.0

    return {
        "iou_per_class": iou_per_class,
        "miou": miou,
        "num_valid_classes": len(ious)
    }


def get_dataset_images(split='test'):
    """
    Liste toutes les images d'un split du dataset Cityscapes.

    Args:
        split: 'test' ou 'val'

    Returns:
        Liste de chemins d'images (triée)
    """
    import os
    from glob import glob

    # Chemin du dataset - utilise sample_data pour Streamlit Cloud
    # ou chemin local si disponible
    local_dataset = "/home/ser/Bureau/Projet_image_new/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
    sample_dataset = "sample_data/images"

    # Préférer le dataset local si disponible, sinon utiliser sample_data
    if os.path.exists(local_dataset):
        dataset_path = local_dataset
        split_path = os.path.join(dataset_path, split)
        pattern = f"{split_path}/**/*_leftImg8bit.png"
    elif os.path.exists(sample_dataset):
        # Utiliser sample_data (ignorer le split car il n'y a qu'un seul dossier)
        pattern = f"{sample_dataset}/*_leftImg8bit.png"
    else:
        return []

    images = glob(pattern, recursive=True)
    return sorted(images)


def load_image_from_path(image_path):
    """
    Charge une image depuis un chemin.

    Args:
        image_path: Chemin vers l'image

    Returns:
        PIL.Image
    """
    return Image.open(image_path)


def plot_class_distribution(distribution):
    """
    Génère un graphique de distribution des classes.

    Args:
        distribution: Dict {class_name: percentage}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(distribution.keys())
    percentages = list(distribution.values())
    colors = [COLOR_PALETTE[i] / 255.0 for i in range(len(classes))]

    bars = ax.barh(classes, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Pourcentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des Classes dans l\'Image', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Ajouter les valeurs sur les barres
    for bar, pct in zip(bars, percentages):
        if pct > 1.0:  # Afficher seulement si > 1%
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


# ============================================================================
# Interface Principale
# ============================================================================

def main():
    global CLASS_NAMES

    # Initialiser le dark mode dans session_state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True  # Dark mode par défaut

    # Appliquer le CSS
    st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

    # En-tête
    st.markdown('<div class="main-header">🚗 Cityscapes Semantic Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Future Vision Transport - Système de Vision pour Véhicules Autonomes</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Dark Mode Toggle
        dark_mode = st.toggle("🌙 Mode Sombre", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()

        st.markdown("---")
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Future+Vision+Transport", use_container_width=True)

        st.markdown("---")
        st.markdown("### ℹ️ À Propos")
        st.markdown("""
        Cette application démontre notre système de segmentation sémantique
        pour véhicules autonomes.

        **Modèle** : ConvNeXt-OCR
        **Framework** : Keras/TensorFlow
        """)

        st.markdown("#### 📊 Performance du Modèle")
        st.markdown("""
        - **Validation mIoU** : **79.21%**
        - **Train mIoU** : 86.73%
        - **Pixel Accuracy** : 94.8%
        - **Architecture** : ConvNeXt-Base + OCR
        - **Classes** : 8 catégories Cityscapes
        """)

        st.markdown("---")
        st.markdown("### 🔧 Paramètres")

        # Affichage de l'URL de l'API (lecture seule)
        st.text(f"URL de l'API : {API_URL}")

        # Test de connexion API
        st.markdown("#### État de l'API")
        is_healthy, health_info = check_api_health()

        if is_healthy:
            st.markdown('<div class="success-box">✅ API opérationnelle</div>', unsafe_allow_html=True)
            with st.expander("Détails"):
                st.json(health_info)
        else:
            st.markdown('<div class="warning-box">⚠️ API non accessible</div>', unsafe_allow_html=True)
            st.error(f"Erreur : {health_info}")
            st.info("Assurez-vous que l'API est démarrée :\n\n`uvicorn api_prediction:app --host 0.0.0.0 --port 8000`")

        st.markdown("---")
        st.markdown("### 🎨 Classes de Segmentation")

        # Afficher les classes
        classes_info = get_api_classes()
        if classes_info:
            for class_data in classes_info["classes"]:
                color_rgb = class_data["color_rgb"]
                color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                    f'<div style="width: 30px; height: 20px; background-color: {color_hex}; '
                    f'border: 1px solid #000; margin-right: 10px;"></div>'
                    f'<span><b>{class_data["id"]}</b> - {class_data["name"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Contenu principal - Tabs pour différentes sources d'images
    tab1, tab2 = st.tabs(["📂 Explorer le Dataset", "📤 Upload Manuel"])

    with tab1:
        st.markdown("## 📂 Parcourir les Images du Dataset")
        st.markdown("Explorez les images de validation avec détection automatique du ground truth.\n\n"
                    "⚠️ **Note**: Les fichiers test de Cityscapes ne contiennent pas les vraies annotations "
                    "(labels masqués pour préserver l'intégrité du benchmark). Utilisez **val** pour voir les ground truths.")

        # Initialiser session_state pour le carousel
        if 'dataset_split' not in st.session_state:
            st.session_state.dataset_split = 'val'
        if 'image_index' not in st.session_state:
            st.session_state.image_index = 0

        col1, col2 = st.columns([1, 3])

        with col1:
            # Sélection du split
            split = st.selectbox(
                "Dataset",
                options=['val', 'test'],
                index=0 if st.session_state.dataset_split == 'val' else 1,
                key='split_selector'
            )

            # Si le split change, réinitialiser l'index
            if split != st.session_state.dataset_split:
                st.session_state.dataset_split = split
                st.session_state.image_index = 0

            # Charger les images du dataset
            dataset_images = get_dataset_images(st.session_state.dataset_split)

            if len(dataset_images) == 0:
                st.warning(f"⚠️ Aucune image trouvée dans le split '{st.session_state.dataset_split}'")
                st.info("Cette fonctionnalité nécessite le dataset Cityscapes en local.")
            else:
                st.success(f"✓ {len(dataset_images)} images disponibles")

                # Navigation
                st.markdown(f"**Image {st.session_state.image_index + 1}/{len(dataset_images)}**")

                col_prev, col_next = st.columns(2)
                with col_prev:
                    if st.button("⬅️ Précédent", disabled=(st.session_state.image_index == 0)):
                        st.session_state.image_index -= 1
                        st.rerun()

                with col_next:
                    if st.button("Suivant ➡️", disabled=(st.session_state.image_index >= len(dataset_images) - 1)):
                        st.session_state.image_index += 1
                        st.rerun()

                # Sélection directe
                selected_index = st.number_input(
                    "Ou sélectionner l'image #",
                    min_value=1,
                    max_value=len(dataset_images),
                    value=st.session_state.image_index + 1,
                    step=1
                )
                if selected_index - 1 != st.session_state.image_index:
                    st.session_state.image_index = selected_index - 1
                    st.rerun()

        with col2:
            if len(dataset_images) > 0:
                # Charger l'image actuelle
                current_image_path = dataset_images[st.session_state.image_index]
                current_image = load_image_from_path(current_image_path)
                image_filename = os.path.basename(current_image_path)

                st.markdown(f"### {image_filename}")
                st.image(current_image, caption=f"{current_image.size[0]}×{current_image.size[1]}", use_container_width=True)

                # Chercher automatiquement le ground truth
                auto_gt = find_ground_truth(image_filename)

                if auto_gt is not None:
                    st.success("✅ Ground truth trouvé")

                    # Coloriser le GT (détection automatique du format)
                    auto_gt_colored = colorize_mask(auto_gt)

                    # Afficher le GT
                    with st.expander("👁️ Voir le Ground Truth"):
                        st.image(auto_gt_colored, caption="Ground Truth (colorisé)", use_container_width=True)

                    # Bouton pour segmenter
                    if st.button("🚀 Segmenter cette image", type="primary", use_container_width=True, key=f"segment_{st.session_state.image_index}"):
                        # Convertir l'image PIL en bytes pour l'API
                        import io
                        img_byte_arr = io.BytesIO()
                        current_image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)

                        # Créer un objet fichier simulé
                        from io import BytesIO
                        class FakeFile:
                            def __init__(self, data, name):
                                self.data = data
                                self.name = name
                                self.type = 'image/png'  # MIME type
                            def getvalue(self):
                                return self.data.getvalue()
                            def seek(self, pos):
                                return self.data.seek(pos)
                            def read(self, size=-1):
                                return self.data.read(size)

                        fake_file = FakeFile(img_byte_arr, image_filename)

                        # Appel API - obtenir le masque brut (non colorisé)
                        success, result = predict_segmentation(fake_file, return_colored=False, return_raw_mask=True)

                        if success:
                            st.markdown("---")
                            st.markdown("### 🎯 Résultats de la Segmentation")

                            # Calculer les métriques IoU
                            # Décoder le masque base64 (PNG image)
                            pred_mask_pil = decode_base64_to_image(result['mask_base64'])
                            pred_mask_array = np.array(pred_mask_pil)
                            gt_mask_array = np.array(auto_gt)

                            metrics = calculate_iou_metrics(pred_mask_array, gt_mask_array)

                            # Afficher métriques
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("mIoU", f"{metrics['miou']:.2f}%")
                            with col_m2:
                                # IMPORTANT: Convertir le GT avant de calculer la pixel accuracy
                                gt_for_accuracy = gt_mask_array.copy()
                                if gt_for_accuracy.max() > 7:
                                    mapped_gt = np.zeros(gt_for_accuracy.shape, dtype=np.uint8)
                                    for original_id, category_id in ID_TO_CATEGORY.items():
                                        mapped_gt[gt_for_accuracy == original_id] = category_id
                                    gt_for_accuracy = mapped_gt
                                pixel_acc = (pred_mask_array == gt_for_accuracy).sum() / pred_mask_array.size * 100
                                st.metric("Pixel Accuracy", f"{pixel_acc:.2f}%")
                            with col_m3:
                                st.metric("Classes valides", f"{metrics['num_valid_classes']}/8")

                            # Afficher les images
                            col_r1, col_r2, col_r3 = st.columns(3)
                            with col_r1:
                                st.markdown("**Image Originale**")
                                st.image(current_image, use_container_width=True)
                            with col_r2:
                                st.markdown("**Masque Prédit**")
                                # Coloriser le masque localement
                                colored_mask = colorize_mask(pred_mask_array)
                                st.image(colored_mask, use_container_width=True)
                            with col_r3:
                                st.markdown("**Ground Truth**")
                                # Recoloriser le GT pour l'affichage des résultats (auto-détection du format)
                                gt_display = colorize_mask(auto_gt)
                                st.image(gt_display, use_container_width=True)

                            # IoU par classe
                            st.markdown("### 📊 IoU par Classe")
                            iou_data = []
                            for class_id, iou_val in metrics['iou_per_class'].items():
                                class_name = CLASS_NAMES[class_id]
                                iou_data.append({"Classe": class_name, "IoU (%)": f"{iou_val:.2f}"})

                            st.table(iou_data)

                        else:
                            st.error(f"❌ Erreur lors de la segmentation : {result}")
                else:
                    st.info("💡 Ground truth non trouvé pour cette image")

    with tab2:
        st.markdown("## 📤 Upload d'Image")
        st.markdown("Uploadez une image pour obtenir sa segmentation sémantique.")

        # Zone d'upload
        uploaded_file = st.file_uploader(
            "Choisissez une image (PNG, JPG)",
            type=['png', 'jpg', 'jpeg'],
            help="Sélectionnez une image de scène urbaine"
        )

        if uploaded_file is not None:
            # Afficher l'image originale
            original_image = Image.open(uploaded_file)

            st.markdown("---")
            st.markdown("## 🖼️ Image Originale")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(original_image, caption=f"{uploaded_file.name} ({original_image.size[0]}×{original_image.size[1]})", use_container_width=True)

            # Chercher automatiquement le ground truth
            st.markdown("---")
            st.markdown("### 🎯 Ground Truth (pour calcul IoU)")

            auto_gt = find_ground_truth(uploaded_file.name)

            if auto_gt is not None:
                st.success(f"✅ Ground truth trouvé automatiquement pour {uploaded_file.name}")
                gt_image = auto_gt
                gt_file = "auto"
            else:
                st.info("💡 Ground truth non trouvé automatiquement. Vous pouvez l'uploader manuellement ci-dessous.")
                gt_file = st.file_uploader(
                    "Upload manuel du Ground Truth (PNG)",
                    type=['png'],
                    help="Masque de segmentation réel (avec IDs de classes 0-7)",
                    key="ground_truth"
                )
                if gt_file is not None:
                    gt_image = Image.open(gt_file).convert('L')
                else:
                    gt_image = None

            # Options de prédiction
            st.markdown("---")
            st.markdown("## ⚙️ Options de Prédiction")

            col1, col2 = st.columns(2)
            with col1:
                show_overlay = st.checkbox("Afficher l'overlay", value=True, help="Superpose la segmentation sur l'image originale")
            with col2:
                show_distribution = st.checkbox("Afficher la distribution des classes", value=True)

            # Bouton de prédiction
            if st.button("🚀 Lancer la Segmentation", type="primary", use_container_width=True):

                # Réinitialiser le pointeur du fichier
                uploaded_file.seek(0)

                # Appel API
                success, result = predict_segmentation(uploaded_file, return_colored=True)

                if success:
                    st.markdown("---")
                    st.markdown("## 🎯 Résultats de la Segmentation")

                    # Afficher les métriques
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("État", "✅ Succès")
                    with col2:
                        mask_shape = result["mask_shape"]
                        st.metric("Résolution Masque", f"{mask_shape[1]}×{mask_shape[0]}")
                    with col3:
                        num_classes_present = sum(1 for v in result["class_distribution"].values() if v > 0)
                        st.metric("Classes Détectées", num_classes_present)

                    # Décoder le masque base64
                    mask_base64 = result["colored_mask_base64"]
                    mask_bytes = base64.b64decode(mask_base64)
                    mask_image = Image.open(io.BytesIO(mask_bytes))

                    # Afficher les images
                    st.markdown("### 📊 Visualisations")

                    if show_overlay:
                        # Créer overlay
                        original_array = np.array(original_image.resize(mask_image.size))
                        mask_array = np.array(mask_image)
                        overlay_array = (0.6 * original_array + 0.4 * mask_array).astype(np.uint8)
                        overlay_image = Image.fromarray(overlay_array)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(original_image, caption="Image Originale", use_container_width=True)
                        with col2:
                            st.image(mask_image, caption="Masque de Segmentation", use_container_width=True)
                        with col3:
                            st.image(overlay_image, caption="Overlay", use_container_width=True)
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_image, caption="Image Originale", use_container_width=True)
                        with col2:
                            st.image(mask_image, caption="Masque de Segmentation", use_container_width=True)

                    # Calcul IoU si ground truth fourni
                    if gt_image is not None:
                        st.markdown("---")
                        st.markdown("### 🎯 Métriques IoU (avec Ground Truth)")

                        # Convertir en array
                        gt_array = np.array(gt_image)

                        # Obtenir le masque prédit (IDs de classes, pas colorisé)
                        # On doit reconvertir le masque colorisé en IDs de classes
                        mask_array = np.array(mask_image)

                        # Convertir RGB vers ID de classe
                        pred_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
                        COLOR_PALETTE_NP = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70],
                                                       [102, 102, 156], [190, 153, 153], [153, 153, 153],
                                                       [250, 170, 30], [220, 220, 0]])

                        for class_id in range(8):
                            color = COLOR_PALETTE_NP[class_id]
                            mask_match = np.all(mask_array == color, axis=-1)
                            pred_mask[mask_match] = class_id

                        # Calculer IoU
                        iou_metrics = calculate_iou_metrics(pred_mask, gt_array, num_classes=8)

                        # Afficher mIoU
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 mIoU", f"{iou_metrics['miou']:.2f}%")
                        with col2:
                            st.metric("✅ Classes Valides", iou_metrics['num_valid_classes'])
                        with col3:
                            st.metric("📏 Pixel Accuracy", f"{((pred_mask == gt_array).sum() / gt_array.size * 100):.2f}%")

                        # Afficher IoU par classe
                        with st.expander("📋 IoU par Classe"):
                            for class_id, iou_val in iou_metrics['iou_per_class'].items():
                                if iou_val > 0:
                                    st.write(f"**{CLASS_NAMES[class_id]}** : {iou_val:.2f}%")

                    # Distribution des classes
                    if show_distribution:
                        st.markdown("### 📈 Distribution des Classes")

                        if gt_image is None:
                            st.info("""
                            **ℹ️ Note** : Cette section montre la **distribution des classes prédites** dans l'image.

                            💡 **Pour calculer l'IoU et le mIoU** :
                            - Les images du dataset Cityscapes sont automatiquement détectées
                            - Sinon, uploadez manuellement le Ground Truth ci-dessus

                            Les métriques globales du modèle (79.21% mIoU) sont affichées dans la barre latérale.
                            """)

                        distribution = result["class_distribution"]

                        # Graphique
                        fig = plot_class_distribution(distribution)
                        st.pyplot(fig)

                        # Tableau détaillé
                        with st.expander("📋 Détails de la Distribution"):
                            col1, col2 = st.columns(2)

                            for i, (class_name, percentage) in enumerate(distribution.items()):
                                target_col = col1 if i < 4 else col2
                                with target_col:
                                    color_rgb = COLOR_PALETTE[i]
                                    color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
                                    st.markdown(
                                        f'<div class="metric-card">'
                                        f'<div style="display: flex; align-items: center; justify-content: space-between;">'
                                        f'<div style="display: flex; align-items: center;">'
                                        f'<div style="width: 25px; height: 25px; background-color: {color_hex}; '
                                        f'border: 2px solid #000; margin-right: 10px; border-radius: 3px;"></div>'
                                        f'<b>{class_name.capitalize()}</b>'
                                        f'</div>'
                                        f'<span style="font-size: 1.2rem; font-weight: bold; color: #1f77b4;">'
                                        f'{percentage:.2f}%</span>'
                                        f'</div>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )

                    # Boutons de téléchargement
                    st.markdown("### 💾 Téléchargement")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Télécharger le masque
                        mask_buffer = io.BytesIO()
                        mask_image.save(mask_buffer, format='PNG')
                        mask_buffer.seek(0)

                        st.download_button(
                            label="📥 Télécharger le Masque",
                            data=mask_buffer,
                            file_name=f"mask_{uploaded_file.name}",
                            mime="image/png",
                            use_container_width=True
                        )

                    with col2:
                        # Télécharger le JSON des résultats
                        json_str = json.dumps(result, indent=2)

                        st.download_button(
                            label="📥 Télécharger les Résultats (JSON)",
                            data=json_str,
                            file_name=f"results_{uploaded_file.name}.json",
                            mime="application/json",
                            use_container_width=True
                        )

                else:
                    st.error(f"❌ Erreur lors de la segmentation : {result}")

        else:
            # Message d'accueil
            st.info("👆 Uploadez une image pour commencer la segmentation")

            # Exemples
            st.markdown("---")
            st.markdown("## 📷 Images d'Exemple")
            st.markdown("Vous pouvez tester avec des images du dataset Cityscapes ou vos propres photos de scènes urbaines.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Scène Urbaine**")
                st.markdown("- Routes")
                st.markdown("- Bâtiments")
                st.markdown("- Véhicules")

            with col2:
                st.markdown("**Éléments de Sécurité**")
                st.markdown("- Panneaux")
                st.markdown("- Feux tricolores")
                st.markdown("- Marquages au sol")

            with col3:
                st.markdown("**Infrastructure**")
                st.markdown("- Trottoirs")
                st.markdown("- Poteaux")
                st.markdown("- Clôtures")


    # ============================================================================
    # Footer
    # ============================================================================

def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p><b>Future Vision Transport</b> - Système de Vision pour Véhicules Autonomes</p>
            <p>Modèle : ConvNeXt-OCR | Performance : 79.21% mIoU | Framework : Keras/TensorFlow</p>
            <p>© 2024 Future Vision Transport. Tous droits réservés.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# Point d'Entrée
# ============================================================================

if __name__ == "__main__":
    main()
    add_footer()
