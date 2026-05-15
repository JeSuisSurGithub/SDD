# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd

# ------------------------ 
from pandas.api.types import is_numeric_dtype

def normalisation(df):
    for column in df.columns:
        if is_numeric_dtype( df[column].dtype ):
            minCol = df[column].min()
            maxCol = df[column].max()
            df[column] = (df[column] - minCol) / (maxCol- minCol)
    return df

from abc import ABC, abstractmethod

class Distance(ABC):
    """ Classe abstraite pour représenter des mesures de distances
        Elle permet de définir une hiérarchie pour les distances
    """
    def __init__(self,nom):
        """ Constructeur:
            prend en argument le nom (str) de la distance créé
        """
        self.__nom:str = nom
        
    @abstractmethod
    def calcule(self, v, M):
        """ Arguments:
                - v: un vecteur 
                - M: un vecteur ou une matrice 
            Hypothèse: v et M ont le même nombre de colonnes
            Retour:
                - un float si M est une vecteur: distance entre v et M
                - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        # le calcul de distance dépend de la mesure que l'on utilise, il sera implémenté
        # dans les sous-classes de cette classe.
        pass
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return "Distance "+self.__nom

class DistanceEuclidienne(Distance):
    """ Classe représentant la distance euclidienne
    """
    def __init__(self):
        """ Constructeur
        """
        super().__init__("euclidienne")
        
    def calcule(self, v, M):
        """ Arguments:
                - v: un vecteur 
                - M: un vecteur ou une matrice
            Hypothèse: v et M ont le même nombre de colonnes
            Retour:
                - un float si M est une vecteur: distance entre v et M
                - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        if v.ndim != 1:
            raise TypeError("Argument incorrect: le premier argument doit être un vecteur")

        if len(M.shape) == 1:
            return np.linalg.norm((M - v).to_numpy(), axis=0)

        e = (M-v)**2
        return np.sqrt(np.sum((M-v)**2, axis=1))
        
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return super().__str__()

from abc import ABC, abstractmethod

class Linkage(ABC):
    """ Classe abstraite pour représenter des approches Linkage
    """
    def __init__(self,nom):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
        """
        self.__nom: str = nom
        
    @abstractmethod
    def calcule(self, G1, G2, verbose= False):
        """ Arguments:
                - G1 et G2 sont des dataframes ou des np.array
                - verbose: pour afficher des messages de débuggage si besoin
            Hypothèse: 
                - G1 et G2 ont le même nombre de colonnes
            Retour:
                - la distance entre G1 et G2 selon le linkage
        """
        pass
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return "Linkage "+self.__nom

class LinkageComplete(Linkage):
    """ Classe pour le linkage "Complete"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("complete")
        self.__distance = distance
        
    def calcule(self, G1, G2,verbose=False):
        """ Arguments:
                - G1 et G2 sont des dataframes ou des np.array
                - verbose: pour afficher des messages de débuggage si besoin
            Hypothèse: 
                - G1 et G2 ont le même nombre de colonnes
            Retour:
                - la distance entre G1 et G2 selon le linkage
        """

        if len(G1.shape) == 1:
            return np.max(self.__distance.calcule(G1, G2))

        if len(G2.shape) == 1:
            return np.max(self.__distance.calcule(G2, G1))
            
        res = -1.0
        for i in range(len(G1)):
            dist = np.max(self.__distance.calcule(G1.iloc[i], G2))
            res = max(res, float(dist))
        return res

        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return super().__str__()+ " ("+self.__distance.__str__()+")"

class LinkageSimple(Linkage):
    """ Classe pour le linkage "Simple"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("simple")
        self.__distance = distance
        
    def calcule(self, G1, G2,verbose=False):

        if len(G1.shape) == 1:
            return np.min(self.__distance.calcule(G1, G2))

        if len(G2.shape) == 1:
            return np.min(self.__distance.calcule(G2, G1))
            
        res = 2**64-1
        for i in range(len(G1)):
            dist = np.min(self.__distance.calcule(G1.iloc[i], G2))
            res = min(res, float(dist))
        return res
        
# ------------------------------------------------------
class LinkageAverage(Linkage):
    """ Classe pour le linkage "Average"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("average")     
        self.__distance = distance
        
    def calcule(self, G1, G2,verbose=False):

        if len(G1.shape) == 1:
            return np.mean(self.__distance.calcule(G1, G2))

        if len(G2.shape) == 1:
            return np.mean(self.__distance.calcule(G2, G1))

        return np.mean(np.array([np.mean(self.__distance.calcule(G1.iloc[i], G2)) for i in range(len(G1))]))

# ------------------------------------------------------
class LinkageCentroide(Linkage):
    """ Classe pour le linkage "Centroide"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("centroide")
        self.__distance = distance    
        
    def calcule(self, G1, G2,verbose=False):

        if len(G1.shape) == 1 and len(G2.shape) == 1:
            return self.__distance.calcule(G1, G2)

        if len(G1.shape) == 1:
            return self.__distance.calcule(G1, np.mean(G2, axis=0))

        if len(G2.shape) == 1:
            return self.__distance.calcule(G2, np.mean(G1, axis=0))

        return self.__distance.calcule(np.mean(G2, axis=0), np.mean(G1, axis=0))

# ------------------------------------------------------
def CHA_initialise(df):
    return {i:[i] for i in range(len(df))}

def CHA_fusionne(df, part, linkage, verbose):
    minDist = 2**64
    part1Cle = None
    part2Cle = None

    p1 = part.copy()
    
    
    for cle1 in part.keys():
        for cle2 in part.keys():
            if cle1 == cle2:
                continue
                
            newDist = linkage.calcule(df.iloc[part[cle1]], df.iloc[part[cle2]], verbose)
            # print(newDist)
            if newDist < minDist:
                minDist = newDist
                part1Cle = cle1
                part2Cle = cle2

    newIndex = max(part.keys())+1
    p1[newIndex] = part[part1Cle] + part[part2Cle]
    p1.pop(part1Cle, None)
    p1.pop(part2Cle, None)
        
    return p1, part1Cle, part2Cle, minDist

def CHA_algorithme(df,linkage,verbose):
    part = CHA_initialise(df)
    info = []
    while len(part) > 1:
        p, c1, c2, d = CHA_fusionne(df, part, linkage, verbose)
        part = p
        info.append([c1, c2, d, len(p[max( p.keys() )])] )
    return info

def CHA_dendrogramme(cha_info, linkage):
    # Paramètre de la fenêtre d'affichage: 
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme: ' + str(linkage), fontsize=25)    
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)
    
    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        cha_info, 
        leaf_font_size=24.,  # taille des caractères de l'axe des X
    )
    
    # Affichage du résultat obtenu:
    plt.show()

class DistanceMinkowski(Distance):
    """ Classe représentant la distance euclidienne
    """
    def __init__(self, p):
        """ Constructeur
        """
        super().__init__("euclidienne")
        self.p = p
        
    def calcule(self, v, M):
        """ Arguments:
                - v: un vecteur 
                - M: un vecteur ou une matrice
            Hypothèse: v et M ont le même nombre de colonnes
            Retour:
                - un float si M est une vecteur: distance entre v et M
                - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        if v.ndim != 1:
            raise TypeError("Argument incorrect: le premier argument doit être un vecteur")

        if len(M.shape) == 1:
            return np.linalg.norm((M - v).to_numpy(), axis=0, ord=self.p)

        return np.pow(np.sum((M-v)**self.p, axis=1), 1/self.p)
        
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return super().__str__()

    