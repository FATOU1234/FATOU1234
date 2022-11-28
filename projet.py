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

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
# from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
# import io
# from pathlib import Path

import pandas as pd
import numpy as np
# import requests
# import toml
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LogisticRegression
import pickle
#DATASET
#df = pd.read_csv('eco2mix-regional-cons-def.csv', sep =',', error_bad_lines=False) 

#Affichage des dix premières lignes.

# display(df.tail(10))
# df.info()
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
    return Image.open(f"{image_name}")


@st.cache(allow_output_mutation=True)
#@st.cache(suppress_st_warning=True)
def fetch_and_clean_data():

    ########################  ECO2MIX  ########################
    #df_eco2mix=pd.read_csv('eco2mix-regional-cons-def.csv', sep =';', error_bad_lines=False)
    df_eco2mix=pd.read_csv("https://drive.google.com/uc?export=download&id=1ETwJ7TnCFs4pd2BIAgI5TUQqTkLChjJ8&confirm=uuid=5e31a1d5-ac6f-46cd-9906-69170e80cbb8", sep =';', error_bad_lines=False)
    df_eco2mix_init=df_eco2mix.head(10)
    
    #Nettoyage des données suite au résultat de la corrélation.
    df_eco2mix['Date']=pd.to_datetime(df_eco2mix['Date'])
    df_eco2mix=df_eco2mix.loc[df_eco2mix['Date'].dt.year>=2016]
    df_eco2mix=df_eco2mix.drop(['Code INSEE région','Date - Heure','Nature', 'Stockage batterie','Déstockage batterie', 'Eolien terrestre', 'Eolien offshore','TCO Thermique (%)', 'TCO Nucléaire (%)', 'TCO Eolien (%)','TCO Solaire (%)',  'TCO Hydraulique (%)', 'TCO Bioénergies (%)','Column 30'], axis=1)
    df_eco2mix=df_eco2mix.drop(['TCH Thermique (%)','TCH Nucléaire (%)','TCH Eolien (%)','TCH Solaire (%)','TCH Hydraulique (%)','TCH Bioénergies (%)'],axis=1)
    df_eco2mix=df_eco2mix.fillna(0)
    df_eco2mix['Année'] = df_eco2mix['Date'].dt.year    
    df_eco2mix['mois']=df_eco2mix['Date'].dt.month
    df_eco2mix['jour']=df_eco2mix['Date'].dt.weekday
    
    ########################  JOURS FERIES  ########################
    ##Téléchargement et affichage du dataset des jours feriés.
    df_jour_f = pd.read_csv("jours_feries_metropole.csv", sep=',')    
    #Nettoyage des données.
    df_jour_f.drop(['annee','zone'],axis=1,inplace=True)

    #Remplacement des jours feriés par 1.
    df_jour_f=df_jour_f.replace(to_replace=['1er janvier' ,'Lundi de Pâques' ,'1er mai' ,'8 mai', 'Ascension','Lundi de Pentecôte', '14 juillet', 'Assomption', 'Toussaint' ,'11 novembre','Jour de Noël'], value=[1,2,3,4,5,6,7,8,9,10,11])
    df_jour_f=df_jour_f.rename({'nom_jour_ferie':'jour_ferie'},axis=1)
    df_jour_f=df_jour_f.rename({'date':'Date'},axis=1)
    df_jour_f['Date']=pd.to_datetime(df_jour_f['Date'], dayfirst=True)
    df_jour_f=df_jour_f.loc[df_jour_f['Date'].dt.year>=2016]
  
    ########################  TEMPÉRATURES  ########################
    #Chargement des données du data set températures.
    temp=pd.read_csv("temperature-quotidienne-regionale.csv", sep=';')

    #Suppression de la variable 'Code INSEE région'.
    temp=temp.drop('Code INSEE région',axis=1)

    #Conversion des dates en format datetime.
    temp['Date']=pd.to_datetime(temp['Date'], dayfirst=True)
    temp=temp.loc[temp['Date'].dt.year>=2016]
    temp=temp.sort_values(by='Date')
  
    return df_eco2mix_init,df_eco2mix, df_jour_f,temp
    
###MAIN### 
st.set_page_config(page_title='Projet Eco2mix',layout="wide")


#st.sidebar.image('LOGO.png', width=180)
st.sidebar.image(load_image("logo.png"), width=180)
st.sidebar.title('Projet Energie Eco2mix')

pages=["Introduction","Exploration du jeu de données","Visualisation","Modélisation","Statistiques et indicateurs","Exploration de donnée externe","Conclusion"]
page = st.sidebar.radio("Sommaire", pages)

st.sidebar.markdown('---')
st.sidebar.write('**Fatoumata Bintou TRAORE**')
st.sidebar.image(load_image("datascientest.png"), width=200)

with st.spinner('Chargement des données des dataframes en cours...'):
    df_eco2mix_init,df,jour_f,temp=fetch_and_clean_data() 
    #st.balloons()
#    st.success('Done!')

st.sidebar.write('BootCamp - Septembre 2022')

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
    st.title("Exploration des données Eco2mix et externes")
    DonneesBrutes = st.checkbox('Données éco2mix brutes', value=True)
    if DonneesBrutes:
        st.write(df_eco2mix_init)

    st.write("Afin de bien comprendre notre jeu de données, nous avons commencé par découvrir ses principales caractéristiques : la taille, les variables, les types et les valeurs manquantes.")
    st.write("Le jeu de données présente 1.9 millions de lignes et 32 colonnes partagées en consommation, production, les taux TCO, TCH et autres variables.")
    st.write("Les variables sont majoritairement de type numérique, ce qui facilitera leur exploitation dans la machine Learning. Les quelques variables catégorielles correspondent à la date de la consommation, la région et le code INSEE.")
    st.write("On remarque également que la consommation maximale enregistré est de l'ordre de 15GW, la moyenne nationale est de 4.5GW et que la moitié des départements consomme moins de 4,1GW (la médiane). L'écart type est important ce qui reflète une consommation hétérogène au niveau des départements.")

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(df.corr(), ax=ax, linewidth=.5)
    st.write(fig)
    with st.expander("Détails"):
        st.write("Afin de commencer le nettoyage de notre jeu de données, nous procédons à la suppression de certaines colonnes. Grace à la fonction de la corrélation entre les différentes variables et la variable cible, nous avons supprimé les variables peu corrélées.")
        st.write("On constate un manque important de données concernant l'énergie produite à partir du nucléaire (825K nans) et celle produite par Pompage (860K nans). Après analyse par région des données, on en déduit que les valeurs manquantes concernent majoritairement les régions qui ne disposent pas de ce type d'infrastructure. Les Nans seront donc remplacés par des zéros.")
        st.write("""Pour le reste des valeurs manquantes, nous avons pris la décision de les remplacer par des zéros au regard de leur nombre infime par rapport au volume total des données.
                    Nous manquons d’autres données démographique et économique de l’ensemble du territoire, nous avons pris la décision de traiter les régions séparément et commencer par la région de l’Ile De France.""")
        st.write("Pour mieux comprendre la tendance de la consommation, nous avons décidé de joindre les data sets traitant la température et les jours fériés. Grace à la fonction date time, nous pourrons ainsi faire des visualisations par les différentes unités de temps.  Les données de data frame de la température commence en 01/01/2016, nous avons dû supprimer toutes les lignes du dataset consommation antérieure à cette date.")

    
    DonneesApresNettoyage= st.checkbox('Données éco2mix après le nettoyage')
    if DonneesApresNettoyage:
        st.write(df.head(10))

    JoursFeries= st.checkbox('Jours fériés en métropole')
    if JoursFeries:
        st.info("La liste des jours fériés en métropole téléchargée à partir de [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/)")
        st.write(jour_f.head(10))
    
    temperatures= st.checkbox("Température quotidienne régionale (depuis janvier 2016)")
    if temperatures:
        st.info("""Ce [jeu de données](https://www.data.gouv.fr/fr/datasets/temperature-quotidienne-regionale-depuis-janvier-2016/) présente les températures minimales, maximales et moyennes quotidiennes (en degré celsius), par région administrative française, du 1er janvier 2016 à aujourd'hui.
                 Il est basé sur les mesures officielles du réseau de stations météorologiques françaises. La mise à jour de ce jeu de données est mensuelle.""")
        st.write(temp.head(10))


if page == "Visualisation":
    st.title("DataViz")
    st.header("")
  
    Production = st.checkbox('Production en MW des différentes énergies par année depuis 2016')
    if Production:
        colors = ['#19D3F3','#FECB52','#636EFA','#AB63FA','#00CC96','#EF553B' ]   
        r = df.groupby(['Année'], as_index = False).agg({'Eolien (MW)':'mean',
                                                    'Solaire (MW)' : 'mean',
                                                    'Hydraulique (MW)' : 'mean',
                                                    'Bioénergies (MW)' : 'mean',
                                                    'Nucléaire (MW)' : 'mean',
                                                    'Thermique (MW)' : 'mean',
                                                    'Nucléaire (MW)' : 'mean'
                                                    })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=r['Année'], y=r['Nucléaire (MW)'], name='Nucléaire', marker_color=colors[4]))
        fig.add_trace(go.Bar(x=r['Année'], y=r['Thermique (MW)'], name='Thermique', marker_color=colors[5]))
        fig.add_trace(go.Bar(x=r['Année'], y=r['Hydraulique (MW)'], name='Hydraulique', marker_color=colors[2]))
        fig.add_trace(go.Bar(x=r['Année'], y=r['Eolien (MW)'], name='Eolien', marker_color=colors[0]))
        fig.add_trace(go.Bar(x=r['Année'], y=r['Solaire (MW)'], name='Solaire', marker_color=colors[1]))
        fig.add_trace(go.Bar(x=r['Année'], y=r['Bioénergies (MW)'], name='Bioénergies', marker_color=colors[3]))
            
        fig.update_layout(barmode='stack', title_text="Production en MW des différentes énergies par année depuis 2016" )
        st.plotly_chart(fig, use_container_width=True)

    Consommation = st.checkbox("Consommation d'énergie en fonction des régions")
    if Consommation:
        st.info("La consommation est élévée en Ile de France (12M d'habitants et PIB 57k€/hab/an) et Auvergne-Rhône-Alpes (8 M d'habitants PIB 31k€/hab/an).")
        dfConsommation=df[['Région','Consommation (MW)']].groupby(['Région'],as_index=False).agg({'Consommation (MW)': 'sum'})
        fig = px.treemap(dfConsommation, path=[px.Constant('France'), 'Région'], values='Consommation (MW)',
                      color='Consommation (MW)', hover_data=['Consommation (MW)'])
        st.write(fig)
        
        # fig = px.sunburst(dfConsommation, path=['Région'], values='Consommation (MW)',
                          # color='Consommation (MW)', hover_data=['Consommation (MW)'])
        # st.write(fig)
        dfConsommation.sort_values(by=['Consommation (MW)'], inplace=True, ascending=False)
        st.write(dfConsommation)
        
    Exportation = st.checkbox("Part de l'exportation de l'énergie  par région")
    if Exportation:
        st.info("Nous remarquons que les régions qui produisent le nucléaire exportent plus d'energies.")
        dfExportation=df.groupby(['Région'],as_index=False).agg({'Nucléaire (MW)':'sum','Ech. physiques (MW)':'sum',
                                                        'Hydraulique (MW)':'sum','Consommation (MW)':'sum','Thermique (MW)':'sum',
                                                        'Eolien (MW)':'sum'})    
        fig=plt.figure(figsize=(20,10))
        barWidth=0.1
        x1=np.array(range(12))
        x2=np.array(range(12))+barWidth
        x3=np.array(range(12))+2*barWidth
        x4=np.array(range(12))+3*barWidth
        x5=np.array(range(12))+4*barWidth
        x6=np.array(range(12))+5*barWidth

        plt.bar(x1, dfExportation['Consommation (MW)'],width = barWidth,label="Consommation");
        plt.bar(x2, dfExportation['Nucléaire (MW)'],width = barWidth,label="Production nucléaire ");
        plt.bar(x3, dfExportation['Hydraulique (MW)'],width = barWidth,label="Production hydraulique");
        plt.bar(x4, dfExportation['Thermique (MW)'],width = barWidth,label="Production thermique");
        plt.bar(x5, dfExportation['Eolien (MW)'],width = barWidth,label="Production éolienne");
        plt.bar(x6, dfExportation['Ech. physiques (MW)'],width = barWidth,label="Exportation/importation d'énergie");
        xlim=[0,12]
        plt.xticks([0, 1,2,3, 4, 5,6, 7,8, 9,10,11],['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté',"Bretagne",
                                                     "Centre-Val de Loire","Grand Est","Hauts-de-France",'Normandie',
                                                    'Nouvelle-Aquitaine','Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",
                                                    "Île-de-France"], rotation=90)
        plt.title("Part de l'exportation de l'énergie  par région",fontsize=16, color='black');
        plt.legend();
        st.write(fig)

    ConsoJoursFeries = st.checkbox("Visualisation de la consommation pendant les jours fériés")
    if ConsoJoursFeries:
        #visualisation de la consommation pendant les jours fériés.
        data=pd.merge(df,jour_f,on='Date',how='left')
        fig=plt.figure(figsize=(10,10))
        sns.barplot(x=data['jour_ferie'], y=data['Consommation (MW)']);
        plt.xticks(np.arange(12), ['jours non ferié','1er janvier' ,'Lundi de Pâques' ,'1er mai' ,'8 mai', 'Ascension',
                                   'Lundi de Pentecôte', '14 juillet', 'Assomption', 'Toussaint' ,'11 novembre',
                                   'Jour de Noël'],rotation=60)

        plt.title("Consommation pendant les jours fériés",fontsize=18, color='black')
        st.write(fig)

if page == "Modélisation":
    st.header("Modélisation")
    st.success("Afin d'éviter le temps d'entraînement des modèles, nous les avons entraîné et enregistré en amont à l'aide du module **pickle**")
    
    temp_IDF=temp[temp["Région"]=='Île-de-France']
    data_IDF=df[df['Région']=='Île-de-France']

    #Merge des deux data set.
    data_IDF=pd.merge(data_IDF,jour_f,on='Date',how='left')

    #Rempacement des nans par des zéros.
    data_IDF=data_IDF.fillna(0)
    data_IDF=data_IDF.groupby(['Date','Région','jour','mois','jour_ferie'],as_index=False).agg({'Consommation (MW)':'sum'})

    #Création d'une nouvelle variable 'Consommation J-7 (MW)'.
    data_IDF['Consommation J-7 (MW)']=data_IDF['Consommation (MW)'].shift(7,axis=0)
    #Création d'une nouvelle variable 'Consommation J-30 (MW)'.
    data_IDF['Consommation J-30 (MW)']=data_IDF['Consommation (MW)'].shift(30,axis=0)

    #Fusion des dataframe consommation et température.
    data_IDF=temp_IDF.merge(right=data_IDF, on=["Date","Région"], how='inner')
   
    #Supression des variables catégorielles.
    data_IDF.drop(["ID","Région",'Date'], axis=1, inplace=True)
    
    #Supression de toutes les lignes contenant des valeurs manquantes :
    data_IDF = data_IDF[(data_IDF.isna().sum(axis=1)) < 1]
    
    #Instanciation de la variable target et les variables explicatives.

    X=data_IDF.drop('Consommation (MW)',axis=1)
    Y=data_IDF['Consommation (MW)']

    #st.write(data_IDF.head(10))
    
    def train_model(estimator, name):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2,shuffle=False,random_state=66)
            
        score_test= estimator.score(X_test,Y_test)
        score_train= estimator.score(X_train,Y_train)
        st.info("Le modèle {} a une accurracy sur l'echantillon d'entraînement : **{}**".format(name,score_train))
        st.info("Le modèle {} a une accurracy sur l'echantillon de test : **{}**".format(name,score_test))
        
        #Affichage du score R2 du modèle sur l'échantillon d'apprentissage par validation croisée.
        st.info("Coefficient de détermination obtenu par validation croisée : **{}**".format(cross_val_score(estimator,X_train,Y_train).mean()))

        #Affichage des prédiction du modèle pour X_test.
        y_pred = estimator.predict(X_test)

        #Affichage du MSE du modèle sur l'ensemble de test.
        st.info("Coefficient MSE  : **{}**".format(mean_squared_error(Y_test, y_pred,squared=False)))

        #Affichage du MAE du modèle sur l'ensemble de test.
        st.info("Coefficient MAE  : **{}**".format(mean_absolute_error(Y_test, y_pred)))
   
        #Affichage du MAPE du modèle sur l'ensemble de test.
        st.info("Coefficient MAPE  : **{}**".format(mean_absolute_percentage_error(Y_test, y_pred)))   
        
        ##Affichage de l'homoscédastité des résidus.
        pred_train = estimator.predict(X_train)
        residus = pred_train - Y_train
        # st.write('Homoscédastité des résidus :',residus.mean())
        
        col1, col2 = st.columns(2)

        with col1:  
            st.markdown("<h5 style='text-align: center;'>Nuage de points entre les prédictions du modèle pour X_test et Y_test</h5>", unsafe_allow_html=True)
            #Affichage du graphique entre entre y_pred et Y_test ainsi que la droite x=y
            fig=plt.figure(figsize=(10,10))
            plt.subplot (211)
            plt.scatter(y_pred, Y_test,color='#980a10')
            plt.plot((Y_test.min(), Y_test.max()), (Y_test.min(), Y_test.max()));        
            st.write(fig)

        with col2:
            st.markdown("<h5 style='text-align: center;'>Nuage des points représentant les résidus en fonction des valeurs de y_train</h5>", unsafe_allow_html=True)
            fig=plt.figure(figsize=(10,10))
            plt.subplot (211)
            plt.scatter(Y_train, residus, color='#980a10', s=15)
            plt.plot((Y_train.min(), Y_train.max()), (0, 0), lw=3, color='#0a5798')
            st.write(fig)

        
        return score_test, score_train
    
    model={ "Choisissez le modèle à appliquer" : None,
            "LinearRegression" : pickle.load(open("LinearRegression.joblib", 'rb')),
            "RandomForestRegressor" : pickle.load(open("RandomForestRegressor.joblib", 'rb')),
            "KNeighborsRegressor" : pickle.load(open("KNeighborsRegressor.joblib", 'rb'))
           }
    mod=st.selectbox("",model.keys())
    
    if model[mod] is not None:
        #st.write(model[mod],mod)
        score_test, score_train = train_model(model[mod],mod)

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
