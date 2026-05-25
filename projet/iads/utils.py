# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Donner class1 = class2 = 0, pour toute les classes
def getSampleDataset(X, Y, class1, class2, size):  
    X02 = X
    Y02 = Y
    if class1 is not None and class2 is not None:
        masque = (Y == class1) + (Y == class2)
        X02 = X[masque]    
        Y02 = np.where(Y[masque] == class1, 1, -1)

    index = np.random.permutation(len(X02)) # mélange des index
    Xr = X02[index[:size]]
    Yr = Y02[index[:size]]
    return Xr, Yr

def genere_dataset_uniform(d, nc, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[array, array]
        Hyp: n est pair
        d: nombre de dimensions de la description
        nc: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    desc = np.random.uniform(binf,bsup,(nc*2, d))
    label = np.array([-1 for i in range(0,nc)] + [+1 for i in range(0,nc)])
    return desc, label


def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nc):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """

    negatif = np.random.multivariate_normal(negative_center, negative_sigma, nc)
    positive = np.random.multivariate_normal(positive_center, positive_sigma, nc)
    
    labelNegative = np.ones((nc)) * (-1)
    labelPositive = np.ones((nc))
    
    return np.concatenate((negatif, positive)), np.concatenate((labelNegative, labelPositive))


def plot2DSet(desc,labels,nom_dataset= "Dataset", avec_grid=True):    
    """ array * array * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille, False sinon
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """

    # Extraction des exemples de classe -1:
    data2_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data2_positifs = desc[labels == +1]

    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', color="red", label='classe -1') # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', color="blue", label='classe +1') # 'x' bleu pour la classe +1
    
    # Informations d'affichage :
    plt.title("data2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()  # Grille: à mettre, ou pas
    
    # Visualisation du résultat
    plt.show()


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """

    positif = desc_set[label_set == 1]
    negatif = desc_set[label_set == -1]

    pi = np.arange(len(positif))
    ni = np.arange(len(negatif))
    np.random.shuffle(pi)
    np.random.shuffle(ni)
    
    train_set = np.concatenate( (positif[pi[:n_pos]], negatif[ni[:n_neg]]), axis=0)
    train_label = np.concatenate( (np.ones(n_pos), np.ones(n_neg) * (-1)))
    
    test_set = np.concatenate( (positif[pi[n_pos:]], negatif[ni[n_neg:]]), axis=0)
    test_label = np.concatenate( (np.ones(len(positif)-n_pos), np.ones(len(negatif)-n_neg) * (-1)))
    
    return (train_set, train_label), (test_set, test_label)

def plot2DTrainTestSet(d_train,l_train, d_test,l_test, nom_dataset= "Dataset", avec_grid=True):    
    """ array * array array * array * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille, False sinon
        la fonction doit utiliser les couleurs suivantes:
        - pour les données d'apprentissage : la couleur 'red' pour la classe -1 et 'blue' pour la +1
        - pour les données de test : la couleur 'jaune' pour la classe -1 et 'verte' pour la +1
    """

    # Extraction des exemples de classe -1:
    data2_negatifs = d_train[l_train == -1]
    # Extraction des exemples de classe +1:
    data2_positifs = d_train[l_train == +1]

    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', color="red", label='classe -1') # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', color="green", label='classe +1') # 'x' bleu pour la classe +1


    # Extraction des exemples de classe -1:
    data2_negatifs = d_test[l_test == -1]
    # Extraction des exemples de classe +1:
    data2_positifs = d_test[l_test == +1]

    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', color="yellow", label='classe -1 test') # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', color="blue", label='classe +1 test') # 'x' bleu pour la classe +1


    
    # Informations d'affichage :
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(avec_grid)  # Grille: à mettre, ou pas

    # Visualisation du résultat
    plt.show()

def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    cov = np.array([[var,0],[0,var]])
    set1, lab1 = genere_dataset_gaussian(np.array([1,0]), cov, np.array([1,1]), cov, n)
    set2,lab2 = genere_dataset_gaussian(np.array([0,1]), cov, np.array([0, 0]), cov, n)
    return np.concatenate((set1,set2),axis=0), np.concatenate((lab1,lab2),axis=0)


def crossval(X, Y, n_iterations, iteration):
    q = len(X) // n_iterations
    Xapp = np.concatenate( (X[: q*iteration], X[q*(iteration+1):]) )
    Yapp = np.concatenate( (Y[: q*iteration], Y[q*(iteration+1):]) )
    
    Xtest = X[q*iteration : q*(iteration+1)]
    Ytest = Y[q*iteration : q*(iteration+1)]

    return Xapp, Yapp, Xtest, Ytest





