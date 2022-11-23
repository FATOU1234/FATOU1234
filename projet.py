# Pour le faire tourner: avoir les sources data dans le même dossier. Ouvrir une commande et taper:
# streamlit run Streamlit.py

import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import datetime as dt
# import seaborn as sns
# from sklearn import model_selection, preprocessing
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
# from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
import io
from pathlib import Path

import pandas as pd
import requests
import toml
from PIL import Image
#DATASET
#df = pd.read_csv('eco2mix-regional-cons-def.csv', sep =',', error_bad_lines=False) 

#Affichage des dix premières lignes.

# display(df.tail(10))
# df.info()

def get_project_root() -> str:
    """Returns project root path.
    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent.parent)

def load_image(image_name: str) -> Image:
    """Displays an image.
    Parameters
    ----------
    image_name : str
        Local path of the image.
    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(Path(get_project_root()) / f"references/{image_name}")

###MAIN###
st.set_page_config(page_title='Projet Eco2mix',layout="wide")

#st.sidebar.image('LOGO.png', width=180)
st.sidebar.title('Projet Energie Eco2mix')

pages=["Introduction","Exploration du jeu de données","Statistiques et indicateurs","Exploration de donnée externe","Conclusion"]
page = st.sidebar.radio("Sommaire", pages)

st.sidebar.markdown('---')
st.sidebar.write('**Fatoumata Bintou TRAORE**')
#st.sidebar.image("datascientest.png", width=200)
st.sidebar.image(load_image("logo.png"), use_column_width=True)



# st.markdown('''
    # <a href="https://datascientest.com">
     # <img src="D:\éCO2mix\Projet_Binette\LOGO.png" alt="Visiter le site MDN">    
    # </a>''',      unsafe_allow_html=True
# )

st.sidebar.write('BootCamp - Septembre 2022')

# col1, col2, col3 = st.columns(3)

# with col1:
   # st.header("A cat")
   # st.image("https://static.streamlit.io/examples/cat.jpg")

# with col2:
   # st.header("A dog")
   # st.image("https://static.streamlit.io/examples/dog.jpg")

# with col3:
   # st.header("An owl")
   # st.image("https://static.streamlit.io/examples/owl.jpg")

if page == "Introduction":    
    st.title("Présentation du projet")
    #st.subheader("Implantation géographique des éoliennes et des stations météorologiques utilisées")
    
    st.write("Il est facile de comprendre que notre économie entière repose en pratique sur la consommation d’énergie. Dans ce cadre, le phasage entre la consommation et la production énergétique au niveau national et au niveau départemental (risque de black out notamment) est nécessaire.")
    st.write("La complexité de ce sujet réside dans le fait que l’énergie électrique n’est pas stockable, d’où l’importance de prédire la consommation.")
    st.info("[La source de données](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure) est celle de l’ODRE (Open Data Réseaux Energies) dans lequel on a accès à toutes les informations de conso et production par filière jour par jour (toutes les 1/2 heure) depuis 2013, du fichier température et du fichier des jours fériés.")
    st.write("Nous avons fait un travail de documentation pour comprendre les différents processus de production mais le document n’est pas d’une grande qualité concernant le renseignement des données")

    st.subheader("Objectif : Constater le phasage entre la consommation et la production énergétique au niveau national et au niveau départemental (risque de black out notamment)")
    st.write("- Analyse au niveau départemental pour en déduire une prévision de consommation")
    st.write("- Analyse par filière de production : énergie nucléaire / renouvelable")
    st.write("- Focus sur les énergies renouvelables (où sont elles implantées)")


if page == "Exploration du jeu de données":    
    st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

    with st.expander("See explanation"):
        st.write("Afin de commencer le nettoyage de notre jeu de données, nous procédons à la suppression de certaines colonnes. Grace à la fonction de la corrélation entre les différentes variables et la variable cible, nous avons supprimé les variables peu corrélées.")
        st.write("On constate un manque important de données concernant l'énergie produite à partir du nucléaire (825K nans) et celle produite par Pompage (860K nans). Après analyse par région des données, on en déduit que les valeurs manquantes concernent majoritairement les régions qui ne disposent pas de ce type d'infrastructure. Les Nans seront donc remplacés par des zéros.")
        st.write("""Pour le reste des valeurs manquantes, nous avons pris la décision de les remplacer par des zéros au regard de leur nombre infime par rapport au volume total des données.
Nous manquons d’autres données démographique et économique de l’ensemble du territoire, nous avons pris la décision de traiter les régions séparément et commencer par la région de l’Ile De France.""")
        st.write("Pour mieux comprendre la tendance de la consommation, nous avons décidé de joindre les data sets traitant la température et les jours fériés. Grace à la fonction date time, nous pourrons ainsi faire des visualisations par les différentes unités de temps.  Les données de data frame de la température commence en 01/01/2016, nous avons dû supprimé toutes les lignes du data set consommation antérieure à cette date.")

if page == "Conclusion":    
    st.title("Conclusion") 
    st.subheader("Perspectives et regard critique")  

    st.write("**Points négatifs:**")
    st.write("Données de mauvaises qualités : (exemple : manque d’unités de mesure, variables non expliqués…)")
    st.write("Manque de données démographique et  socio-économique par région pour une meilleure analyse")
  
    st.write("**Et si on avait 3 mois de plus…**")
    st.write("Analyse des TCH (Taux de charge de la production ) pour prédire les blackout mais présence de nombre important de valeurs manquantes. Nous avons commencé le travail de calcul de TCH pour chaque filière.")
    
    st.write("**Pour conclure :**")    
    st.write("- La production nucléaire est la première source  d’énergie en France")
    st.write("- Les régions productrices du nucléaire sont les régions exportatrices")
    st.write("- La production de l’énergie renouvelable en France est  très faible")
    st.write("- La consommation dépends de plusieurs facteurs : température, jours travaillé ou non et heure de la journée.")
    st.write("- Excepté un évènement exceptionnel,  la consommation reste stable d’une année à l’autre")
    st.write("- Le modèle Random Forest est optimal pour la région Ile de France mais ce n’est pas le cas pour toutes les autres  régions.")


    
    #st.write('Ce modèle final donne des rappels > 75% pour toute catégorie et une accuracy à 83%. Les erreurs de '
    #         'classification se font pour la plupart avec la classe adjacente.')
    #st.write("Paramètres: {'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 50}")
    #st.write('**Matrice de confusion**')
    #
