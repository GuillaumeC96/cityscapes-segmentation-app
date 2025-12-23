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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# URL de l'API (à modifier selon votre déploiement)
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Palette de couleurs Cityscapes
COLOR_PALETTE = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
], dtype=np.uint8)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign'
]

# ============================================================================
# Configuration de la Page
# ============================================================================

st.set_page_config(
    page_title="Cityscapes Segmentation Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
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
""", unsafe_allow_html=True)

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


def predict_segmentation(image_file, return_colored=True):
    """
    Envoie une image à l'API pour prédiction.

    Args:
        image_file: Fichier image uploadé
        return_colored: Si True, retourne le masque colorisé

    Returns:
        dict: Résultats de la prédiction
    """
    try:
        # Préparer le fichier avec le bon format
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        params = {"return_colored": return_colored}

        # Appel API
        with st.spinner("Segmentation en cours..."):
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                params=params,
                timeout=60
            )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Erreur API : {response.status_code} - {response.text}"

    except Exception as e:
        return False, f"Erreur : {str(e)}"


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
        "/home/ser/Bureau/Projet_image_new/P8_Cityscapes_gtFine_trainvaltest/gtFine",  # Dataset officiel
        "/home/ser/Bureau/Projet_image_new/data",  # Dataset augmenté/personnalisé
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
                    return Image.open(matches[0]).convert('L')
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
    # En-tête
    st.markdown('<div class="main-header">🚗 Cityscapes Semantic Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Future Vision Transport - Système de Vision pour Véhicules Autonomes</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
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

    # Contenu principal
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
                        CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign']

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
