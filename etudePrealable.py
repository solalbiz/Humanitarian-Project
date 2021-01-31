#%% Imports

import pandas as pd
import numpy as np

import scipy.cluster.hierarchy as sch
import sklearn.cluster as skc
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import sklearn.mixture as mix
import sklearn.decomposition as skd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from time import time
from functools import wraps


#%% Fonctions d'initialisation et de pré-traitement

def importTestData(filepath):
    """
    

    Parameters
    ----------
    filepath : string
        Chemin du fichier.

    Returns
    -------
    Dataframe
        Dataframe contenant les données.

    """
    df=pd.read_csv(filepath,delimiter='\s+')
    avecResultats=df.shape[1]==3
    if avecResultats:
        df.columns=['c1','c2','resultat']
        resultats=df['resultat']
        df=df.drop('resultat',axis=1)
    data=centrerReduire(df)
    if avecResultats:
        return data, resultats
    else:
        return data

def centrerReduire(df):
    """
    

    Parameters
    ----------
    df : dataframe
        Dataframe à centrer réduire.

    Returns
    -------
    data : dataframe
        Dataframe centrée réduite.

    """
    data=df.to_numpy()
    scaleur=skp.StandardScaler()
    data=scaleur.fit_transform(data)
    return data

#%% Mesures et exploitation

def mesureTps(fonction):
    """
    

    Parameters
    ----------
    fonction : function
        Fonction dont on veut mesurer le temps d'execution
    Returns
    -------
    float
        Temps d'execution.

    """
    @wraps(fonction)
    def wrapper(*args, **kwargs):
        start=time()
        res=fonction(*args,**kwargs)
        end=time()
        t=end-start
        if t>1:
            print("temps : ",t)
        return res
    return wrapper

def tailleClusters(prediction):
    """
    

    Parameters
    ----------
    prediction : numpy array
        L'array contenant les prédictions du clustering.

    Returns
    -------
    Serie
        Série du nombre d'elements par cluster.

    """
    return pd.Series(prediction).value_counts()

def tailleClusterBruit(prediction):
    """
    

    Parameters
    ----------
    prediction : numpy array
         L'array contenant les prédictions du clustering.

    Returns
    -------
    int
        Utile uniquement avec le dbscan.

    """
    taille=tailleClusters(prediction)
    if -1 in taille:
        return taille[-1]
    else:
        return 0
    
def nombreClusters(prediction):
    """
    

    Parameters
    ----------
    prediction : numpy array
        L'array contenant les prédictions du clustering.

    Returns
    -------
    int
        Nombre de clusters.

    """
    return len(np.unique(prediction))
    

def score(prediction, data=None, vraiesValeurs=None):
    """
    

    Parameters
    ----------
    prediction : numpy array
        L'array contenant les prédictions du clustering.
    data : dataframe, optional
        Dataframe contenant les données. The default is None.
    vraiesValeurs : numpy array, optional
        Les vrais valeurs de clustering. The default is None.

    Returns
    -------
    s : float
        Score (Silhouette si il n'y a pas de vraies valeurs, et Rand sinon).

    """
    tc=tailleClusters(prediction)
    print('Taille des clusters : ', tc)
    if (vraiesValeurs is None and data is not None):
        s=skm.silhouette_score(data, prediction)
        print('silhouette-score : ', s)
    if (vraiesValeurs is not None):
        s=skm.adjusted_rand_score(vraiesValeurs, prediction)
        print('rand : ',s)
    return s

#%% Algorithmes

@mesureTps
def kmeans(data,k,  algo='elkan', plot=False, dimensionPlot=2):
    """
    

    Parameters
    ----------
    data : dataframe
        Données normées.
    k : int
        Nombre de clusters.
    algo : string, optional
        Algorithme utilisé pour le clustering. The default is 'elkan'.
    plot : boolean, optional
        Représentation graphique ou non. The default is False.
    dimensionPlot : int, optional
        Pour faire un plot en 2D ou 3D. The default is 2.

    Returns
    -------
    predict : numpy array
        L'array contenant les prédictions du clustering.

    """
    km=skc.KMeans(k,init='k-means++',algorithm=algo,n_init=100)
    km.fit(data)
    predict=km.predict(data)
    if plot:
        centers = km.cluster_centers_
        affiche(data,predict,centers=centers, dimension=dimensionPlot)
    return predict


@mesureTps
def gaussian(data,k, plot=False, dimensionPlot=2):
    """
    

    Parameters
    ----------
    data : dataframe
        Données normées.
    k : int
        Nombre de clusters.
    plot : boolean, optional
        Représentation graphique ou non. The default is False.
    dimensionPlot : int, optional
        Pour faire un plot en 2D ou 3D. The default is 2.

    Returns
    -------
    predict : numpy array
        L'array contenant les prédictions du clustering.

    """
    gm=mix.GaussianMixture(n_components=k,n_init=100)
    gm.fit(data)
    predict=gm.predict(data)
    return predict

@mesureTps
def cha(data, t, z=None, methode='ward', metrique='euclidean', plot=False,dimensionPlot=2):
    """
    

    Parameters
    ----------
    data : dataframe
        Données normées.
    t : float
        Inertie inter-cluster.
    z : float, optional
        Resultat du linkage. The default is None.
    methode : string, optional
        Methode de calcul de distance inter-cluster. The default is 'ward'.
    metrique : string, optional
        Type de distance utilisé. The default is 'euclidean'.
    plot : boolean, optional
        Représentation graphique ou non. The default is False.
    dimensionPlot : int, optional
        Pour faire un plot en 2D ou 3D. The default is 2.

    Returns
    -------
    predict : numpy array
        L'array contenant les prédictions du clustering.

    """
    if z is None:
        z=sch.linkage(data, method=methode , metric=metrique, optimal_ordering=plot)
    predict=sch.fcluster(z, t, criterion='distance')
    if plot:
        sch.dendrogram(z)
        affiche(data,predict, dimension=dimensionPlot)
    return predict

@mesureTps
def dbscan(data, epsilon, nVoisins=5, metrique='minkowski',p=2, plot=False,dimensionPlot=2): 
    """
    

    Parameters
    ----------
    data : dataframe
        Données normées.
    epsilon : float
        Rayon du voisinage observé pour chaque point.
    nVoisins : int, optional
        Nombre de voisins. The default is 5.
    metrique : string, optional
        Type de distance utilisé. The default is 'minkowski'.
    p : int, optional
        Paramètre de la métrique de Minkowski. The default is 2.
    plot : boolean, optional
        Représentation graphique ou non. The default is False.
    dimensionPlot : int, optional
        Pour faire un plot en 2D ou 3D. The default is 2.

    Returns
    -------
    predict : numpy array
        L'array contenant les prédictions du clustering.

    """
    predict=skc.dbscan(data,eps=epsilon,  min_samples=nVoisins, metric=metrique, p=2)[1]
    if plot:
        affiche(data,predict, dimension=dimensionPlot)
    return predict


#%% Affichage

def affiche(data, resultat=None, dimension=2, centers=None):
    """
    

    Parameters
    ----------
    data : dataframe
        Données normées.
    resultat : TYPE, optional
        Clustering. The default is None.
    dimension : int, optional
        2D ou 3D. The default is 2.
    centers : numpy array, optional
        Centre des clusters. The default is None.

    Returns
    -------
    None.

    """
    fig = plt.figure()   
    if data.shape[1]>=3 and dimension==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Axe 1')
        ax.set_ylabel('Axe 2')
        ax.set_zlabel('Axe 3')
        ax.scatter(data[:,0],data[:,1],data[:,2], c=resultat)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:2], c='black', s=200, alpha=0.5);
    else:
        ax=fig.add_subplot(111)
        ax.scatter(data[:,0],data[:,1],c=resultat)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    fig.show()

def afficheFrame(df,colonneAbscisses,colonneOrdonnee, prediction=None):
    """
    

    Parameters
    ----------
    df : dataframe
        Données normées.
    colonneAbscisses : string
        Nom de la colonne en abscisse.
    colonneOrdonnee : string
        Nom de la colonne en ordonnée.
    prediction : numpy array, optional
        Prediction du clustering. The default is None.

    Returns
    -------
    None.

    """
    plt.scatter(df[colonneAbscisses], df[colonneOrdonnee], c=prediction)
    plt.xlabel(colonneAbscisses)
    plt.ylabel(colonneOrdonnee)
    plt.show()

def afficheFrameComplete(df, prediction=None):
    """
    

    Parameters
    ----------
    df : Dataframe
        Données normées.
    prediction : numpy array, optional
        Prediction du clustering. The default is None.

    Returns
    -------
    None.

    """
    for i in range(len(df.columns)):
        for j in range(i):
           afficheFrame(df, df.columns[i], df.columns[j], prediction)


#%% Appels aux fonctions

if __name__=="__main__":
    dataAggregation,resultatsAggregation=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/Aggregation.txt')
    dataG2_2_20=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/g2-2-20.txt')
    dataG2_2_100=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/g2-2-100.txt')
    dataJain,resultatsJain=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/jain.txt')
    dataPathbased,resultatsPathbased=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/pathbased.txt')
    affiche(dataAggregation)
    affiche(dataG2_2_20)
    affiche(dataG2_2_100)
    affiche(dataJain)
    affiche(dataPathbased)

    