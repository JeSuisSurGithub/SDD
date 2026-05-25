# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy
from iads import utils as ut
import time
import matplotlib.pyplot as plt

# ------------------------ 


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.std(L)


def validation_croisee(C, DS, nb_iter, verbose = False):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
        Arguments:
            - C (Classifieur): un classifieur déjà défini (mais pas entraîné) 
            - DS (tuple[array,array]: un tuple composé d'un dataset (data, labels)
            - nb_iter (int): nombre d'itérations à réaliser
            - verbose: pour dire si on veut afficher des messages au cours de l'exécution
        Retour: tuple[ list[float], float, float]
            - triplet contenant la liste des performances obtenues, la performance moyenne et l'écart type
    """
    X, Y = DS
    
    # 1) mélanger des exemples 
    index = np.random.permutation(len(X)) # mélange des index
    Xm = X[index]
    Ym = Y[index]
    perf = []
    
    for i in range(nb_iter):
        
        Xapp, Yapp, Xtest, Ytest = ut.crossval(Xm, Ym, nb_iter, i)
        classifieur = copy.deepcopy(C)
        classifieur.train(Xapp, Yapp)
        perf.append(classifieur.accuracy(Xtest, Ytest))

    taux_moyen, taux_ecart = analyse_perfs(perf)
    
    if verbose:
        print(f'Analyse perf : moyenne: {taux_moyen:0.4f}\tecart: {taux_ecart:0.4f}')

    return perf, taux_moyen, taux_ecart

def matrice_confusion(reverser, y_pred: np.ndarray, y_test: np.ndarray):
    """
    Retourne la précision de la prédiction et affiche la matrice de confusion

    Parameters:
    - reverser: Liste des classes
    - y_pred (np.ndarray): Classes prédites
    - y_test (np.ndarray): Vraie classes

    Returns:
    - accuracy (float): Taux de succès
    """
    accuracy = np.mean(y_test == y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.hist2d(y_test, y_pred, bins=[10, 10])
    plt.title('Matrice de Confusion')
    plt.text(0.05, 0.95, f'Précision: {accuracy:.2f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.xticks(np.unique(y_test), reverser)
    plt.xlabel('Vrai Classe')
    plt.yticks(np.unique(y_pred), reverser)
    plt.ylabel('Classe Prédite')
    plt.colorbar()
    
    plt.savefig("mat.png")
    plt.show()
    plt.close()