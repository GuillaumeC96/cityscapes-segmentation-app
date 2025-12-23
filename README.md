# Cityscapes Segmentation - Application Streamlit

Application web de segmentation sémantique pour véhicules autonomes utilisant ConvNeXt-OCR.

## 🎯 Fonctionnalités

- Upload d'images (drag-and-drop)
- Segmentation sémantique en temps réel
- Visualisation : Image originale / Masque / Overlay
- Distribution des classes (graphique)
- Téléchargement des résultats (PNG + JSON)

## 🚀 Démarrage Rapide

```bash
# Installer les dépendances
pip install -r requirements_streamlit.txt

# Lancer l'application
streamlit run app_streamlit.py
```

## 📦 Configuration

L'application se connecte à l'API de prédiction via `.streamlit/secrets.toml`.

Pour l'utiliser avec votre propre API, modifiez ce fichier :

```toml
API_URL = "http://votre-api-url:8000"
```

## 🌐 Déploiement sur Streamlit Cloud

1. Fork ce repository
2. Allez sur https://share.streamlit.io
3. Connectez-vous avec GitHub
4. Sélectionnez ce repository
5. Main file path: `app_streamlit.py`
6. Ajoutez dans "Advanced settings > Secrets":
   ```toml
   API_URL = "http://13.60.240.14:8000"
   ```
7. Déployez !

## 📊 Performance du Modèle

- **mIoU Validation**: 79.21%
- **Architecture**: ConvNeXt-Base + OCR
- **Classes**: 8 catégories Cityscapes
- **Temps d'inférence**: ~2.5s/image (GPU)

## 🔗 Liens

- API Documentation: http://13.60.240.14:8000/docs
- Dataset: [Cityscapes](https://www.cityscapes-dataset.com)

## 📄 Licence

Projet académique - Future Vision Transport
