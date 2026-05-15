# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2026

# Import de packages externes
import numpy as np
import pandas as pd
import graphviz as gv

import copy

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    #### A compléter pour répondre à la question posée
    classes, nb = np.unique(Y, return_counts=True)
    return classes[np.argmax(nb)]

def shannon(P,k=2):
    """ list[Number] * int -> float
        Hypothèse: P est une distribution de probabilités et k>0
        - P: distribution de probabilités
        - k: base du logarithme à utiliser, par défaut 2
        rend la valeur de l'entropie de Shannon correspondante
    """
    ########################## COMPLETER ICI 
    P = np.array(P)
    if k != 0 and k != 1:
        return -np.sum(P[P != 0] * np.log(P[P != 0]) / np.log(k) )
    return 0.0
    # return -np.sum(P[P != 0] * np.log2(P[P != 0]) )
    
    ##########################

def entropie(Y, k=2):
    """ Y : (array) : ensemble de labels de classe
        Hypothèse: k>0
        - k: base du logarithme à utiliser, par défaut 2
        rend l'entropie de l'ensemble Y
    """
    ########################## COMPLETER ICI 
    classes, nb = np.unique(Y, return_counts=True)
    nbTot = np.sum(nb)
    probas = nb / nbTot

    k = len(nb)
    return shannon(probas, k)

    ##########################


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.__attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.__nom_attribut = 'att_'+str(num_att)
        else:
            self.__nom_attribut = nom 
        self.__Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.__classe   = None       # valeur de la classe si c'est une feuille

    def getattribut(self):
        return self.__attribut
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.__Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.__Les_fils == None:
            self.__Les_fils = dict()
        self.__Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.__classe    = classe
        self.__Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple 
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.__classe
        if exemple[self.__attribut] in self.__Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.__Les_fils[exemple[self.__attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.__nom_attribut,' -> Valeur inconnue: ',exemple[self.__attribut])
            return None
    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.__Les_fils:
            total += self.__Les_fils[noeud].compte_feuilles()
        return total
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.__classe),shape='box')
        else:
            g.node(prefixe, self.__nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.__Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), str(valeur))
                i = i+1        
        return g

    

from abc import ABC, abstractmethod

# ---------------------------
# ------------------------ A COMPLETER :
# importation de la librairie abc:
from abc import ABC, abstractmethod

class Classifier(ABC):
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    __nombre_crees: int = 0  # Variable de classe pour compter le nombre de classifiers créés
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__nombre_crees += 1
        self.__ident = Classifier.__nombre_crees  # identifiant du classifieur (unique)
        self.__dimension = input_dimension
        
    def get_dimension(self):
        """ Accesseur de la variable __dimension 
        """
        return self.__dimension
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f'Classifier #{self.__ident} (d{self.__dimension})'
        
    @abstractmethod
    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        pass

    @abstractmethod
    def score(self,x) -> float:
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
    
    @abstractmethod
    def predict(self, x) -> int:
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pass

    def accuracy(self, desc_set, label_set) -> float:
        """ rend le taux d'exemples bien classés dans le dataset
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        return np.mean(label_set==np.apply_along_axis(self.predict, 1, desc_set))

# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        self.__k= k
        # les 2 variables suivantes seront utilisées dans la méthode train()
        self.__desc_set= None   
        self.__labels_set= None

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f" {self.get_dimension(), self.__k}"

    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.__desc_set = desc_set
        self.__labels_set = label_set

    def score(self,x) -> float:
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un array
        """
        
        tabDist = np.linalg.norm(self.__desc_set-x, axis=1)
        ind = np.argsort(tabDist)
        kppv = self.__labels_set[ind][:self.__k]
        p = np.mean(kppv == 1)
        score = 2*(p-0.5)
        return score

    
    def predict(self, x) -> int:
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un array
        """
        return 1 if self.score(x) > 0 else -1
    
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        v = np.random.uniform(-1, 1, input_dimension)
        norm = np.linalg.norm(v)
        self.__w  = v / norm
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f"Classifier #24 (d2) - LinAleatoire w= {self.__w}"
     
    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur !")

    def score(self,x) -> float:
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un array
        """
        return np.dot(x, self.__w.T)
    
    def predict(self, x) -> int:
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un array
        """
        return 1 if self.score(x) > 0 else -1

    
class ClassifierKNN_MC(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self,input_dimension,k,nc):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
                - nc (int): nombre de classes
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.__k= k
        # les 2 variables suivantes seront utilisées dans la méthode train()
        self.__desc_set= None   
        self.__labels_set= None
        self.__nc = nc
    
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f" {self.get_dimension(), self.__k, self.__nc}"
    
    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.__desc_set = desc_set
        self.__labels_set = label_set
    
    def score(self,x) -> float:
        """ rend un vecteur avec le nombre de voisin de chaque classe
        """
        # Calculer le tableau des distances entre x et les points du set
        tab_dist = np.linalg.norm(self.__desc_set-x, axis=1)
        ind = np.argsort(tab_dist)[:self.__k]
        
        kppv = self.__labels_set[ind]
        unique, counts = np.unique(kppv, return_counts=True)
        return unique[np.argmax(counts)]
        
    def predict(self, x) -> int:
        """ rend la prediction sur x
            x: une description : un array
        """
        return self.score(x)
        
class ClassifierPerceptronTME3(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
                - verbose: pour dire si on veut afficher la valeur d'initialisation
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        self.__learning = learning_rate
        if init:
            self.__w = np.zeros(input_dimension)
        else:
            self.__w = 0.001 * (2 * np.random.uniform(size=input_dimension) - 1 )

        self.__allw =[self.__w.copy()]
        
        if verbose:
            print(f"{super().__str__()} (TME3): initialisation (learning rate= {self.__learning}) w= {self.__w}")

        
    def train_step(self, desc_set, label_set, stabilised=False):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: array avec des descriptions
                - label_set: array avec les labels correspondants
        """        
        vi = np.arange(len(desc_set))
        np.random.shuffle(vi)
        for i in vi:
            xi = desc_set[i]
            yi = label_set[i]
            yi_ = self.score(xi)
            predit = 1 if yi_ > 0 else -1
            # print(self.__w, yi, xi, predit)
            if (label_set[i] != predit and stabilised == False) or (yi_ * yi < 1 and stabilised == True):
                self.__w = self.__w + self.__learning * (yi- yi_) * xi
                self.__allw.append(self.__w.copy())

    def get_allw(self):
        return self.__allw

    def get_w(self):
        return self.__w
    
    def set_w(self, w):
        self.__w = w
    

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f"(TME3) {self.get_dimension(), self.__learning, self.__w}"

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001, stabilised=False, verbose=False):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: array avec des descriptions
                - label_set: array avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
                - verbose (par défaut: False): affichage de messages d'information
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """  
        nb_iter = 0
        vd = []
        for _ in range(nb_max):
            wm = self.__w.copy()

            self.train_step(desc_set, label_set)
            nb_iter += 1
    
            delta = np.linalg.norm(np.abs(self.__w-wm))
            vd.append(delta)
            if delta < seuil:
                break
        
        if verbose:
            print(f"train {self} : {nb_iter} appels à train_step")
        return vd
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.__w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        return 1 if self.score(x) > 0 else -1

    
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
                - verbose: pour dire si on veut afficher la valeur d'initialisation
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        self.__learning = learning_rate
        if init:
            self.__w = np.zeros(input_dimension+1) 
        else:
            self.__w = 0.001 * (2 * np.random.uniform(size=input_dimension) - 1 )
            self.__w = np.append(self.__w, (np.random.rand()))
            
        self.__allw =[self.__w.copy()]
        
        if verbose:
            print(f"{super().__str__()}: initialisation (learning rate= {self.__learning}) w= {self.__w}")

    def augmente(v):
        if len(v.shape) < 2:
            return np.append(v, -1)
        else:
            return np.append(v, np.ones((v.shape[0], 1))*-1, axis=1)
        
    def train_step(self, desc_set, label_set, stabilised=False):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: array avec des descriptions
                - label_set: array avec les labels correspondants
        """        
        vi = np.arange(len(desc_set))
        np.random.shuffle(vi)
        for i in vi:
            xi = desc_set[i]
            yi = label_set[i]
            yi_ = self.score(xi)
            predit = 1 if yi_ > 0 else -1
            # print(self.__w, yi, xi, predit)
            if (label_set[i] != predit and stabilised == False) or (yi_ * yi < 1 and stabilised == True):
                self.__w = self.__w + self.__learning * (yi- yi_) * ClassifierPerceptron.augmente(xi)
                self.__allw.append(self.__w.copy())
                # print(self.__allw)

    def get_allw(self):
        return self.__allw
    
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f" {self.get_dimension(), self.__learning, self.__w}"

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001, stabilised=False ,verbose=False):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: array avec des descriptions
                - label_set: array avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
                - verbose (par défaut: False): affichage de messages d'information
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """  
        nb_iter = 0
        vd = []
        for _ in range(nb_max):
            wm = self.__w.copy()

            self.train_step(desc_set, label_set, stabilised)
            nb_iter += 1
    
            delta = np.linalg.norm(np.abs(self.__w-wm))
            vd.append(delta)
            if delta < seuil:
                break
        
        if verbose:
            print(f"train {self} : {nb_iter} appels à train_step")
        return vd
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(ClassifierPerceptron.augmente(x), self.__w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        return 1 if self.score(x) > 0 else -1

class ClassifierPerceptronStable(ClassifierPerceptron):
    """ Perceptron de Rosenblatt stabilisé
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True,verbose=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w:
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
                - verbose: pour dire si on veut afficher la valeur d'initialisation
        """
        super().__init__(input_dimension, learning_rate, init,verbose)  # Appel du constructeur de la classe mère
        if verbose:
            print(f"{super().__str__()} (stabilise)")

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        ################### A COMPLETER
        return super().__str__()

    
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001,stabilised=True, verbose=False):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: array avec des descriptions
                - label_set: array avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
                - verbose (par défaut: False): affichage de messages d'information
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        super().train(desc_set, label_set, nb_max, seuil,True, verbose)

class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, input_dimension, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.__cl_bin = cl_bin
        self.__nCl = 0
        self.__lCl = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.__nCl = len(np.unique(label_set))
        for i in range(self.__nCl):
            cli = copy.deepcopy(self.__cl_bin)
            label_tmp = 2 * ((label_set == i) - 0.5)
            cli.train(desc_set, label_tmp)
            self.__lCl.append(cli)
            
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        scores = []
        for cl in self.__lCl:
            scores.append(cl.score(x))
        return scores
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.argmax(self.score(x))