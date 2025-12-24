import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# CONFIGURATION DES CHEMINS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modele_final_prediction_prix.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

st.set_page_config(page_title="ImmoTunis - Grand Tunis", page_icon="🏡", layout="centered")

# CHARGEMENT DES FICHIERS
@st.cache_resource
def load_assets():
    # Charge le modèle et le scaler
    m = joblib.load(MODEL_PATH)
    s = joblib.load(SCALER_PATH)
    return m, s

try:
    model, scaler = load_assets()
    all_features = model.feature_names_in_
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.info("Assurez-vous que les fichiers .pkl sont dans le même dossier que ce script.")
    st.stop()

# --- VOTRE LISTE COMPLÈTE DU GRAND TUNIS ---
GRAND_TUNIS_LISTE = sorted([
    'Tunis', 'Lafayette', 'Mutuelleville', 'Montplaisir', 'Centre Urbain Nord',
    'El Menzah', 'El Menzah 1', 'El Menzah 5', 'El Menzah 6', 'Ennasr', 
    'El Omrane', 'El Omrane Supérieur', 'El Ouardia', 'El Khadra', 
    'Bab Souika', 'Bab Bhar', 'La Kasbah', 'La Medina',
    'Sijoumi', 'Séjoumi', 'Cité Olympique', 'Cité Jardins', 'Bardo', 'Le Bardo',
    'Ariana', 'Ariana Ville', 'Ariana Superieur', 'Ghazela', 'Cite Ghazela',
    'Raoued', 'Raoued Plage', 'Borj Louzir', 'La Soukra', 'Sokra',
    'Chotrana 1', 'Chotrana 2', 'Chotrana 3', 'Ennkhilet',
    'La Marsa', 'Marsa', 'Sidi Daoud', 'Gammarth', 'Gammarth Supérieur',
    'Gammarth Village', 'Cité des Pins', 'Sidi Bou Said',
    'Carthage', 'Carthage Byrsa', 'Carthage Salambo',
    'Le Kram', 'Krame', 'Lac 1', 'Lac 2', 'Les Berges du Lac', 'Jardins de Carthage',
    'Ben Arous', 'Mourouj 1','Mourouj 2','Mourouj 3','Mourouj 4','Mourouj 5','Mourouj 6',
    'Megrine', 'Megrine Jawhara', 'Megrine Chaker', 
    'Rades', 'Rades Plage', 'Ezzahra', 'Boumhel', 'Fouchana', 
    'Hammam Lif', 'Hammam Chatt', 'Medina El Jadida',
    'Manouba', 'Oued Ellil', 'Den Den', 'Mornaguia', 'Douar Hicher'
])

#INTERFACE UTILISATEUR
st.title("🏡 Estimateur Immobilier Grand Tunis")
st.write("Entrez les caractéristiques pour obtenir une estimation du prix de vente.")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        superficie = st.number_input("Superficie habitable (m²)", min_value=20, max_value=800, value=120)
        nb_pieces = st.slider("Nombre de pièces", 1, 10, 3)
        
    with col2:
        ville_selected = st.selectbox("Localisation (Ville/Région)", GRAND_TUNIS_LISTE)

st.markdown("---")

# CALCUL DE LA PRÉDICTION 
if st.button("💰 Calculer l'estimation", use_container_width=True):
    if ville_selected:
        # 1. Créer un DataFrame vide avec les colonnes du modèle
        input_df = pd.DataFrame(0.0, index=[0], columns=all_features)
        
        # 2. Remplir les valeurs numériques
        input_df['superficie'] = superficie
        input_df['nb_pieces'] = nb_pieces
        
        # 3. Appliquer le Scaler (Standardisation)
        input_df[['superficie', 'nb_pieces']] = scaler.transform(input_df[['superficie', 'nb_pieces']])
        
        # 4. Encodage de la ville
        col_ville = f"ville_region_{ville_selected.lower()}"
        
        if col_ville in all_features:
            input_df[col_ville] = 1.0
        
        # 5. Prédiction
        log_pred = model.predict(input_df)
        prix_final = np.expm1(log_pred)[0]
        
        # Affichage du résultat 
        st.balloons()
        st.success("### Estimation Terminée")
        st.metric(label="Prix de Vente Estimé", value=f"{prix_final:,.0f} TND")
        
        st.info(f"Localisation prise en compte : {ville_selected}")
    else:
        st.error("Veuillez sélectionner une ville dans la liste.")
