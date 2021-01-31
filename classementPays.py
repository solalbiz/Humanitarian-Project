from etudePrealable import *

#%% Ouverture des données et définition des variables globales


"""------------------------ MODIFIEZ LE CHEMIN D'ACCES CI-DESSOUS ------------------------"""


df=pd.read_csv('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/data.csv', index_col=0)
    
pays=df.index
colonnes=df.columns


#%% Prétraitement

def remplaceValeurs():
    """
    Remplace les valeurs nulles ou aberrantes dans les données initiales

    Returns
    -------
    None.

    """
    
    df.at["Bangladesh", "life_expectation"]=72.05
    df.at["Australia", "GDP"]=54348.23
    df.at["United Kingdom", "GDP"]=42378.606
    df.at["United States", "GDP"]=65253.518
    df.at['France','total_fertility']=1.92
    df.at['Niger','total_fertility']=7
    df.at['Italy','GDP']=34480
    df.at['Norway','GDP']=81700
    df.at['Nigeria',"inflation"]=13
    

def paysMin(df=df):
    """
    Affiche le min et son index de chaque colonne de la dataframe

    Parameters
    ----------
    df : Dataframe, optional
        Source des donnees. The default is df.

    Returns
    -------
    None.

    """
    for colonne in df.columns:
        print(colonne, df[colonne].min(), df[colonne].idxmin())
        
def paysMax(df=df):
    """
    Affiche le max et son index de chaque colonne de la dataframe

    Parameters
    ----------
    df : Dataframe, optional
        Source des donnees. The default is df.

    Returns
    -------
    None.

    """
    for colonne in df.columns:
        print(colonne, df[colonne].max(), df[colonne].idxmax())

remplaceValeurs()
data=centrerReduire(df)

 
#%% Itérations des algorithmes pour en rechercher les paramètres optimaux

@mesureTps
def itereKmeans(data,vraiesValeurs=None, kMin=2,kMax=10, algo='elkan'):
    """
    Itération de l'algorithme des kmeans pour différentes valeurs de k, et affichage des résultats

    Parameters
    ----------
    data : np.array
        Données centrées réduites.
    vraiesValeurs : np.array, optional
        Résultat attendu. The default is None.
    kMin : int, optional
        nombre de clusters minimal. The default is 2.
    kMax : int, optional
        nombre de clusters maximal. The default is 10.
    algo : 'string', optional
        Algorithme de kmeans à utiliser. The default is 'elkan'.

    Returns
    -------
    None.

    """
    K=list(range(kMin,kMax+1))
    S=[score(kmeans(data, k, algo=algo, plot=False),data,vraiesValeurs) for k in K]
    plt.plot(K,S)
    plt.xLabel("K")
    plt.yLabel("Score")

@mesureTps
def itereGaussian(data,vraiesValeurs=None, kMin=2, kMax=10):
    """
    Itération de l'algorithme des Gaussiennes pour différentes valeurs de k et affichage des résultats

    Parameters
    ----------
    data : np.array
        Données centrées réduites.
    vraiesValeurs : np.array, optional
        Résultat attendu. The default is None.
    kMin : int, optional
        nombre de clusters minimal. The default is 2.
    kMax : int, optional
        nombre de clusters maximal. The default is 10.

    Returns
    -------
    None.

    """
    K=list(range(kMin,kMax+1))
    S=[score(gaussian(data,k,plot=False),data, vraiesValeurs) for k in K]
    plt.plot(K,S)
    plt.xLabel("K")
    plt.yLabel("Score")

@mesureTps
def itereCha(data,vraiesValeurs=None, pasT=0.1,kMin=2, kMax=10,tMin=0.1,tMax=100,methode='ward', metrique='euclidean'):
    """
    Itération de l'algorithme de clustering ascendant pour différentes valeurs de t et affichage des résultats

    Parameters
    ----------
    data : np.array
        Données centrées réduites.
    vraiesValeurs : TYPE, optional
        DESCRIPTION. The default is None.
    pasT : TYPE, optional
        DESCRIPTION. The default is 0.1.
    kMin : TYPE, optional
        DESCRIPTION. The default is 2.
    kMax : TYPE, optional
        DESCRIPTION. The default is 10.
    tMin : TYPE, optional
        DESCRIPTION. The default is 0.1.
    tMax : TYPE, optional
        DESCRIPTION. The default is 100.
    methode : TYPE, optional
        DESCRIPTION. The default is 'ward'.
    metrique : TYPE, optional
        DESCRIPTION. The default is 'euclidean'.

    Returns
    -------
    None.

    """
    nombreDonnees=data.shape[0]
    z=sch.linkage(data, method=methode , metric=metrique, optimal_ordering=True)
    T=np.arange(tMin, tMax, pasT)
    S, K=[],[]
    for t in T:
        pr=cha(data, t, z, methode=methode, metrique=metrique,plot=False)
        kt=nombreClusters(pr)
        if kt>=kMin and kt<=kMax and kt<nombreDonnees:
            S.append(score(pr,data,vraiesValeurs))
            K.append(kt)
        elif kt<kMin:
            break
        else:
            S.append(np.nan)
            K.append(np.nan)
                
    S+=[np.nan for i in range(len(S),len(T))]
    K+=[np.nan for i in range(len(K),len(T))]
    
    plt.plot(T,S)
    plt.xlabel("t")
    plt.ylabel("Score")
    plt.show()
    plt.plot(T,K)
    plt.xlabel("t")
    plt.ylabel("Nombre de clusters")
    plt.show()
    
@mesureTps
def itereDbscan(data,vraiesValeurs=None, pasEpsilon=0.001,kMin=2, kMax=10,epsilonMin=0.01, epsilonMax=5,nombreVoisinsMin=5,nombreVoisinsMax=5, metrique='minkowski', pMetrique=2):
    nombreDonnees=data.shape[0]
    E=np.arange(epsilonMin,epsilonMax,pasEpsilon) 
    S,K,B=[],[],[]
    for nv in range(nombreVoisinsMin,nombreVoisinsMax+1):
        Sn,Kn,Bn=[],[],[]
        for epsilon in E:
            pr=dbscan(data,epsilon=epsilon, nVoisins=nv, metrique=metrique, p=pMetrique, plot=False)
            kne=nombreClusters(pr)
            if kne>=kMin and kne<nombreDonnees and kne<=kMax:
                Sn.append(score(pr,data,vraiesValeurs))
                Kn.append(kne)
                Bn.append(tailleClusterBruit(pr))
            # elif kne<kMin:
            #         break
            else:
                Sn.append(np.nan)
                Kn.append(np.nan)
                Bn.append(np.nan)
                
        Sn+=[np.nan for i in range(len(Sn),len(E))]
        Kn+=[np.nan for i in range(len(Kn),len(E))]
        Bn+=[np.nan for i in range(len(Bn),len(E))]
        
        S.append(Sn)
        K.append(Kn)
        B.append(Bn)
    
    
    for sn in S:
        plt.plot(E,Sn)
    plt.show()
    
    for kn in K:
        plt.plot(E,kn)
    # plt.xLabel("Epsilon")
    # plt.yLabel("Nombre de clusters")
    plt.show()
    
    for bn in B:
        plt.plot(E,bn)
    # plt.xLabel("Epsilon")
    # plt.yLabel("Taille du cluster considéré comme du bruit")
    plt.show()
    
    
#%% ACP

def acp(i,data=data):
    pca=skd.PCA(n_components=i)
    return pca.fit_transform(data)
    

#%% Analyse des clusters et selection des pays

def indexesCluster(prediction, cluster=None):
    if cluster is None:
        u=np.unique(prediction)
        return {x:(np.where(prediction==x)[0]) for x in u}
    else:
        return np.where(prediction==cluster)[0]
    
def paysCluster(prediction, cluster=None, sousSelection=True):
    """Pays associés à chaque cluster
    Paramètres :
        renvoie directement la liste des pays
    Sous
    
    sinon, renvoie un dictionnaire { cluster : tableau des pays appartenant au cluster }
    Auquel cas on peut accéder aux pays en faisant dictionnaire[Cluster]
    
    Les données renvoyées par le knn_search sont des indices parmi les pays passés en paramètre, qui sont des pays du cluster. cette fonction compense ce défaut
    """
    if cluster is None:
        u=np.unique(prediction)
        return {x:(pays[prediction==x])[sousSelection] for x in u}
    else:
        return (pays[prediction==cluster])[sousSelection]

def centreCluster(prediction, cluster, data=data):
    """renvoie le centre du cluster défini par la prédiction"""
    return (data[prediction==cluster]).mean(0)
    
    
def paysFictif(prediction, cluster):
    c=centreCluster(prediction, cluster, data)
    dataCluster=data[prediction==cluster]
    minDataCluster=dataCluster.min(0)
    maxDataCluster=dataCluster.max(0)
    paysF=[maxDataCluster[0], c[1],c[2],c[3],minDataCluster[4],maxDataCluster[5],minDataCluster[6], maxDataCluster[7],minDataCluster[8]]
    return paysF


norms = { "L1":lambda x: np.sum(np.abs(x)),
	  "L2":lambda x: np.sum(x**2),
	  "inf":lambda x: np.max(np.abs(x))
	}

def compute_distances(data, query, norm="L2"):
	"""
    Cette fonction nous a été fournie par M. TIRILLY
    
    Compute distances.

	Computes the distances between the vectors (rows) of a dataset and a
	single query). Three distances are supported:
	  * Manhattan distance ("L1");
	  * squared Euclidean distance ("L2");
	  * Chebyshev distance ("inf").

	:param data: Dataset matrix with samples as rows.
	:param query: Query vector
	:type data: (n,d)-sized Numpy array of floats
	:type query: (d)-sized Numpy array of floats

	:result: The distances of the data vectors to the query.
	:rtype: (n)-sized Numpy array of floats
	"""
	norm_function = norms[norm]
	distances = np.zeros((len(data),), dtype=np.float32)
	for i, d in enumerate(data):
		distances[i] = norm_function(d-query)
	return distances

def knn_search(data, query, k=1, norm='L2'):
	""" Brute-force k-NN search

	Performs a brute-force k-NN search for the given query in data.
	Three distance are supported:
	  * Manhattan distance ("L1");
	  * squared Euclidean distance ("L2");
	  * Chebyshev distance ("inf").

	:param data: Dataset matrix with samples as rows.
	:param query: Query vector
	:param k: Number of nearest neighbors to return
	:param norm: Distance to use ("L1", "L2" (default) or "inf")
	:type data: (n,d)-sized Numpy array of floats
	:type query: (d)-sized Numpy array of floats
	:type k: int
	:type norm: str

	:return: k nearest neighbors (as their indices in the input matrix),
		distances to the query
	:rtype: (k)-sized Numpy array of ints, (k)-sized Numpy array of floats
	"""
	distances = compute_distances(data, query, norm)
	if k == 1:
		min_idx = np.argmin(distances)
		return [min_idx], [distances[min_idx]]
	else:
		min_idx = np.argpartition(distances, k)[:k]
		return min_idx, distances[min_idx]

def indicePaysMin(colonne):
    return df[colonne].idxmin('index')

def indicePaysMax(colonne):
    return df[colonne].idxmax('index')

extremums={'min':indicePaysMin, 'max':indicePaysMax}

caractéristiquesDéterminantes={"child_mortality":'max',
                                 'income':'min', 
                                 'life_expectation':'min', 
                                 'total_fertility':'max', 
                                 'GDP':'min'}

def clusterExtremumColonne(prediction, colonne, extremum):
    """Renvoie le cluster satisfaisant la caractéristique détermnante pour l'attribution de l'aide
    Par exemple Haiti étant le pays ayant la child_mortality la plus haute,
    clusterExtremumColonne(prediction, "child_mortality","max" ) renverra le numero du cluster de Haiti"""
    return prediction[list(pays).index(extremums[extremum](colonne))]

def clustersParCaracteristiques(prediction):
    """Renvoie un dictionnaire {caractéristique : cluster} où cluster est le cluster comprenant le pays nécessitant de l'aide pour une caracteristique"""
    return {colonne : clusterExtremumColonne(prediction,colonne,caractéristiquesDéterminantes[colonne]) for colonne in caractéristiquesDéterminantes}

def clusterSatisfaisantLePlusDeCaracteristiques(prediction):
    """Renvoie le numéro du cluster satisfaisant le plus de caractéristiques nécessitant de l'aide pour une prédiction donnée"""
    serieCompte=pd.Series(clustersParCaracteristiques(prediction)).value_counts()
    if -1 in serieCompte:
        serieCompte.drop(labels=[-1], inplace=True) #On ignore le cluster bruité si besoin
    return serieCompte.idxmax() 

def selectionPays(data, prediction, nombrePaysMax=10, cluster=None):
    if cluster==None:
        cluster=clusterSatisfaisantLePlusDeCaracteristiques(prediction)
    indexesC=indexesCluster(prediction, cluster)
    if len(indexesC)>nombrePaysMax:
        paysF=paysFictif(prediction,cluster)
        indicesTrouves=knn_search(data[indexesC],paysF,nombrePaysMax)[0]
        return paysCluster(prediction,cluster,indicesTrouves) 
    else: 
        return paysCluster(pr,cluster)
    
    

def intersection(listeDeListes):
    """Renvoie l'intersection des éléments des listes dont la liste est passée en paramètre"""
    if len(listeDeListes)==1:
        return set(listeDeListes[0])
    else:
        m=int(len(listeDeListes)/2)
        l1=listeDeListes[:m]
        l2=listeDeListes[m:]
        return intersection(l1) & intersection(l2)

#%% Selection après ACP

def clusterSelon1Dimension(data, prediction, axe=0,extremum='min'):
    u=np.unique(prediction)
    u=u[u!=-1] # pour ignorer le cluster bruité si besoin
    d={}
    for cluster in u:
        d[cluster]=centreCluster(prediction, cluster)[axe]
    #d={ cluster:centreCluster(prediction, cluster)[axe] for cluster in u}
    projetesCentres=pd.Series(d)
    if extremum=='min':
        return projetesCentres.idxmin()
    else:
        return projetesCentres.idxmax()


def paysFictifACP(dimensionACP, prediction, cluster):
    acp=skd.PCA(dimensionACP)
    acp.fit(data)
    return acp.transform([paysFictif(prediction, cluster)])

def selectionPaysACP(data, prediction, axe=0, extremum='min', nombrePaysMax=10,cluster=None):
    if cluster is None:
        cluster=clusterSelon1Dimension(data, prediction,axe, extremum)
    indexesC=indexesCluster(prediction,cluster)
    if len(indexesC)>nombrePaysMax:
        paysF=paysFictifACP(data.shape[1], prediction, cluster)
        indicesTrouves=knn_search(data[indexesC],paysF,nombrePaysMax)[0]
        return paysCluster(prediction,cluster,indicesTrouves)
    else: 
        return paysCluster(prediction,cluster)
    


#%% Résolution du problème
    
def paysDonneesInitiales():
    predictionKmeans=kmeans(data,k=4)
    predictionGaussian=gaussian(data,k=5)
    predictionCHA=cha(data=data, t=25)
    predictionDBSCAN=dbscan(data=data, epsilon=1.24, nVoisins=3)
    
    
    paysKmeans=selectionPays(data, predictionKmeans)
    paysGaussian=selectionPays(data, predictionGaussian)
    paysCHA=selectionPays(data, predictionCHA)
    paysDBSCAN=selectionPays(data, predictionDBSCAN)
    
    return intersection([paysKmeans,paysGaussian,paysCHA]) # On ignore le dbscan car élimine trop de pays

def paysDonneesProjetees():
    projection3=acp(3)
    predictionKmeans=kmeans(projection3,k=3)
    predictionGaussian=gaussian(projection3,k=3)
    predictionCHA=cha(projection3, t=15)
    predictionDBSCAN=dbscan(projection3, epsilon=0.6, nVoisins=4)
    
    paysKmeans=selectionPaysACP(projection3, predictionKmeans, axe=0, extremum='min',nombrePaysMax=10)
    paysGaussian=selectionPaysACP(projection3, predictionGaussian,axe=0, extremum='min',nombrePaysMax=10)
    paysCHA=selectionPaysACP(projection3, predictionCHA,axe=0, extremum='min')
    paysDBSCAN=selectionPaysACP(projection3, predictionDBSCAN,axe=0, extremum='min')

    return intersection([paysKmeans,paysGaussian,paysCHA])

if __name__=='__main__':   
    
    print("Pensez à modifier le chemin d'accès du fichier")
    
    print("--------------------------------------------------------")
    print("                     RESULTATS                          ")
    print("--------------------------------------------------------\n \n")
    print("Examen des donnees :\n")
    
    print("Nb de lignes et colonnes ",df.shape,"\n","nb d'elements ",df.size, "\n")
    print("type de donnees ",df.dtypes)
    print("\n")
    
    print("\n valeurs minimales (sans les valeurs aberrantes")
    print(paysMin(df))
    
    print("\n valeurs maximales (sans les valeurs aberrantes)")
    print(paysMax(df))
    
    df.hist(column="GDP")
    
    print("\n Données manquantes : ")
    print(df.isna().sum())
    
    print("\n")
    
    
    print("\n \n Calculs des Pays:\n")
    
    print("Calcul des pays ayant besoin d'aide humanitaire : données initiales")
    print("Pays à aider déterminés à l'aide des données initiales : \n",paysDonneesInitiales())
    
    
   # print("\n \n Calcul des pays ayant besoin d'aide humanitaire : projection sur 3 axes")
  #  print("Pays à aider déterminés à l'aide des données projetées : \n",paysDonneesProjetees())
    
    
    