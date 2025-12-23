# Application Web Streamlit - Segmentation Cityscapes

Interface web interactive pour tester l'API de segmentation sémantique.

## Fonctionnalités

- **Upload d'images** : Interface drag-and-drop
- **Visualisation** : Image originale, masque, overlay
- **Analyse** : Distribution des classes avec graphiques
- **Téléchargement** : Résultats en PNG et JSON
- **Responsive** : Interface adaptative

## Installation

### 1. Installer les dépendances

```bash
pip install -r requirements_streamlit.txt
```

### 2. Configuration

Modifier le fichier `.streamlit/secrets.toml` pour configurer l'URL de l'API :

```toml
# Développement local
API_URL = "http://localhost:8000"

# Production
API_URL = "https://your-api-url.com"
```

## Démarrage

### Prérequis

L'API doit être démarrée **avant** de lancer l'application :

```bash
# Terminal 1 : Démarrer l'API
uvicorn api_prediction:app --host 0.0.0.0 --port 8000

# Terminal 2 : Démarrer Streamlit
streamlit run app_streamlit.py --server.port 8501
```

### Accès

L'application sera accessible à : `http://localhost:8501`

## Utilisation

### 1. Vérifier l'État de l'API

Dans la sidebar gauche, vérifiez que l'API est opérationnelle (indicateur vert "✅ API opérationnelle").

### 2. Upload d'Image

- Cliquez sur "Browse files" ou glissez-déposez une image
- Formats supportés : PNG, JPG, JPEG
- L'image s'affichera automatiquement

### 3. Configuration

Cochez les options souhaitées :
- **Afficher l'overlay** : Superpose la segmentation sur l'image
- **Afficher la distribution** : Graphique de distribution des classes

### 4. Lancer la Segmentation

Cliquez sur le bouton "🚀 Lancer la Segmentation"

### 5. Résultats

Les résultats affichent :
- **Métriques** : État, résolution, nombre de classes
- **Visualisations** : Originale, masque, overlay
- **Distribution** : Graphique et tableau détaillé
- **Téléchargement** : Masque PNG et résultats JSON

## Structure de l'Application

```
app_streamlit.py
├── Configuration
│   ├── API_URL
│   ├── COLOR_PALETTE
│   └── CLASS_NAMES
├── Fonctions Utilitaires
│   ├── check_api_health()
│   ├── get_api_classes()
│   ├── predict_segmentation()
│   └── plot_class_distribution()
└── Interface Principale
    ├── Header
    ├── Sidebar (API status, classes)
    ├── Upload Zone
    ├── Prediction Options
    ├── Results Display
    └── Footer
```

## Déploiement Cloud

### Option Recommandée : Streamlit Cloud (GRATUIT)

Streamlit Cloud est la solution recommandée pour déployer le frontend de l'application.

#### Avantages
- **Gratuit** pour les applications publiques
- Déploiement automatique depuis GitHub
- Configuration des secrets simplifiée
- Mises à jour automatiques à chaque commit
- HTTPS inclus

#### Étapes de Déploiement

1. **Pusher le code sur GitHub**

```bash
# Créer un repository GitHub
gh repo create cityscapes-segmentation --public

# Pusher le code
git add .
git commit -m "Deploy Streamlit application"
git push origin main
```

2. **Déployer sur Streamlit Cloud**

- Aller sur [share.streamlit.io](https://share.streamlit.io)
- Se connecter avec GitHub
- Cliquer sur "New app"
- Sélectionner le repository : `cityscapes-segmentation`
- Configurer :
  - **Main file path** : `Cassez_Guillaume_3_application_Streamlit_122020/app_streamlit.py`
  - **Python version** : 3.10
  - **Requirements file** : `Cassez_Guillaume_3_application_Streamlit_122020/requirements_streamlit.txt`

3. **Configurer les Secrets**

Dans les "Advanced settings > Secrets" de Streamlit Cloud :

```toml
# Secrets Streamlit Cloud
API_URL = "https://cityscapes-api-env.eu-west-1.elasticbeanstalk.com"
```

**Note** : Remplacer l'URL par celle de votre API déployée sur AWS.

4. **Déployer**

Cliquer sur "Deploy" et attendre la fin du build.

L'application sera accessible à : `https://username-cityscapes-segmentation.streamlit.app`

### Option 2 : Heroku

```bash
# Créer un Procfile
echo "web: streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Créer setup.sh pour configurer Streamlit
cat > setup.sh << 'EOF'
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
EOF

# Déployer
heroku create myapp-streamlit
git push heroku main
```

### Option 3 : Azure Web App

```bash
# Créer un App Service
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name cityscapes-demo \
  --runtime "PYTHON:3.10"

# Configurer les variables d'environnement
az webapp config appsettings set \
  --name cityscapes-demo \
  --resource-group myResourceGroup \
  --settings API_URL="https://your-api.azurewebsites.net"

# Déployer
az webapp up --name cityscapes-demo --resource-group myResourceGroup
```

### Option 4 : Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt

COPY app_streamlit.py .
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build et Run** :
```bash
# Build
docker build -t cityscapes-streamlit .

# Run
docker run -p 8501:8501 -e API_URL="http://api-container:8000" cityscapes-streamlit
```

## Personnalisation

### Modifier le Thème

Éditer `.streamlit/config.toml` :

```toml
[theme]
primaryColor = "#FF4B4B"        # Couleur principale
backgroundColor = "#FFFFFF"      # Fond
secondaryBackgroundColor = "#F0F2F6"  # Fond secondaire
textColor = "#262730"           # Texte
font = "sans serif"             # Police
```

### Ajouter un Logo

Remplacer l'URL du placeholder dans `app_streamlit.py` :

```python
st.image("path/to/your/logo.png", use_column_width=True)
```

### Modifier les Classes Affichées

Les classes sont automatiquement récupérées depuis l'API. Pour modifier les couleurs, éditer `COLOR_PALETTE` dans `app_streamlit.py`.

## Performance

- **Temps de réponse** : Dépend de l'API (~2-3s avec GPU)
- **Taille max upload** : 200MB par défaut (configurable)
- **Sessions simultanées** : Illimité (gratuit sur Streamlit Cloud)

## Troubleshooting

### Erreur : "API non accessible"

```bash
# Vérifier que l'API est démarrée
curl http://localhost:8000/health

# Vérifier l'URL dans .streamlit/secrets.toml
cat .streamlit/secrets.toml
```

### Erreur : "Connection refused"

- L'API n'est pas démarrée
- L'URL de l'API est incorrecte
- Problème de firewall/CORS

### Performance lente

- Vérifier la latence réseau vers l'API
- Optimiser les images avant upload (resize)
- Utiliser un GPU pour l'API

### Layout cassé

- Effacer le cache : `streamlit cache clear`
- Redémarrer l'application
- Vérifier la version de Streamlit

## Sécurité

Pour la production :

1. **Authentification** : Ajouter un système de login
2. **Rate Limiting** : Limiter les requêtes par utilisateur
3. **HTTPS** : Utiliser un certificat SSL
4. **Validation** : Vérifier les fichiers uploadés

Exemple d'authentification basique :

```python
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    main()
```

## Support

Pour toute question ou problème :
- Email : support@futurevision-transport.com
- Documentation API : http://localhost:8000/docs

## Licence

© 2024 Future Vision Transport. Tous droits réservés.
