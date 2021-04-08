from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA

import csv


def Init(fichier):
    """ Construction des données A et b en fonction d'un fichier d'entrée donné en argument """
    with open(fichier, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            A.append([1, int(row[0].split(',')[0])])
            b.append(float(row[0].split(',')[1]))


def E(w):
    return LA.norm(np.matmul(A, np.array(w)) - b)

def gradient(w):
    """Calcul du gradient de E"""
    return np.matmul(Q, np.array(w)) - c


####################################
# Critère d'arrêt de l'algorithme du gradient.
####################################
def stop(eps, grad):
    """ Condition d'arrêt """
    return LA.norm(grad) < eps


####################################
# Pas de Polyak
####################################
def pas(w, k, d):
    return (1 / k)


##########################################################
# Algorithme de gradient à pas consatant
##########################################################
def gradient_descent_a_pas_constant(w0, max_iters):
    eps = 0.0001
    w = w0
    k = 1
    grad = gradient(w)
    while not stop(eps, grad) and k <= max_iters:
        w = w - pas(w, k, -grad) * (grad / LA.norm(grad))
        grad = gradient(w)
        k = k + 1
    return w

####################################
# Pas pour l'algorithme de gradient à forte pente argmin
####################################
def pas_optimal(w):
    grad = gradient(w)
    return ((1 / LA.norm(grad)) * (
                np.matmul(c.transpose(), grad) + np.matmul(np.matmul(np.array(w).transpose(), Q), grad))) * (
                       (LA.norm(grad) ** 2) / (np.matmul(np.matmul(np.array(grad).transpose(), Q), grad)))


##########################################################
# Algorithme de gradient à forte pente
##########################################################
def gradient_descent_a_forte_pente(w0):
    w = w0
    grad = gradient(w)
    prev_w = np.ones(D)
    while not np.array_equal(prev_w, w):
        pas = pas_optimal(w)
        prev_w = w
        w = w - pas * (grad / LA.norm(grad))
        grad = gradient(w)
    return -w


##########################################################
# Gradient en w pour l'algorithme de dichotomie
##########################################################
def derive_g_de_alpha(alpha, w):
    grad = gradient(w)
    return 1 / (LA.norm(grad) ** 2) * (
                -LA.norm(grad) * np.matmul(np.matmul(np.array(w).transpose(), Q), grad) + alpha * np.matmul(
            np.matmul(np.array(grad).transpose(), Q), grad)) - (1 / LA.norm(grad)) * np.matmul(c.transpose(), grad)

##########################################################
# Algorithme de dichotomie
##########################################################
def dichotomie(a, b):
    stop = False
    while a < b and not stop:
        m = (a + b) / 2
        if derive_g_de_alpha(m, w0) == 0:
            stop = True
        elif derive_g_de_alpha(a, w0) * derive_g_de_alpha(m, w0) < 0:
            b = m
        else:
            a = m
    return m

##########################################################
# Détermination du pas alpha
##########################################################
def determiner_alpha(w):
    a = 0
    b = 0
    h = 0.001
    while derive_g_de_alpha(a, w) * derive_g_de_alpha(b, w) > 0:
        if (derive_g_de_alpha(a, w) > 0):
            a = a - h
        else:
            b = b + h
    if (derive_g_de_alpha(a, w) == 0):
        alpha = a
    elif (derive_g_de_alpha(b, w) == 0):
        alpha = b
    else:
        alpha = dichotomie(a, b)
    return alpha

##########################################################
# Gradient descent à pas optimale en utilisant la méthode de dichotomie
##########################################################
def gradient_descent_dichotomie(w0):
    w = w0
    grad = gradient(w)
    prev_w = np.ones(D)
    while not np.array_equal(prev_w, w):
        pas = determiner_alpha(w)
        prev_w = w
        w = w - pas * (grad / LA.norm(grad))
        grad = gradient(w)
    return -w


##########################################################
# Méthode permettant d'afficher le nuage de points lu dans le fichier
# "train_v0.csv" (ou train.csv)
##########################################################
def points():
    plt.plot(np.array(A)[:, 1], b, 'o')


##########################################################
# Méthode affichant la droite de regression correspondant
# aux paramètres optimaux renvoyés par la méthode gradient
##########################################################
def graphique(w):
    min = np.min(np.array(A)[:, 1]) - 2
    max = np.max(np.array(A)[:, 1]) + 2
    x = np.linspace(min, max, 100)
    y = w[1] * x + w[0]
    plt.plot(x, y, '-r')
    plt.show()


#########################################################
# Programme principal
#########################################################
# A est la matrice utilisée dans la fonction d'erreur E(w). Voir énoncé TP
A = []
# b est le vecteur utilisé dans la fonction d'erreur E(w). Voir énoncé TP
b = []

# Lecture des données se trouvant dans le fichier train_v0.csv et stockage dans A et b
Init("train_v0.csv")

# Affichage des contenus de A et de b
print("A = ", A)
print("b = ", b)

D = len(A[0]) - 1  # Nombre de paramètres.
# Initialisation d'un point de départ quelconque
w0 = [0, 0]

# Transformation des listes A et b en objets utilsables par NumPy
A = np.array(A)
b = np.array(b)

# Calculer la matrice Q et le vecteur c en supposant A et b générés.
Q = np.matmul(A.transpose(), A)
c = np.matmul(A.transpose(), b)


#######################################################################"
# "gradient_descent" est la méthode de gradient partant du vecteur de paramètre w0
# Elle retourne une approximation du vecteur de paramètres optimaaux.


''' Méthode à pas constante avec 4000 comme nombre d'itération '''
# k = 1
# wapprox = gradient_descent_a_pas_constant(w0, 4000)

''' Méthode à forte pente '''
#k = 2
#wapprox = gradient_descent_a_forte_pente(w0)

''' Méthode à pas optimal avec la méthode de dichotomie '''
k = 3
wapprox = gradient_descent_dichotomie(w0)


# On affiche ici ce vecteur
print(" wapprox = ", wapprox)
# On affiche ici l'erreur correspondante
print(" E(wapprox) = ", E(wapprox))

# Affichage du nuage de points sur lequel la régression est faite.
if k == 1:
    plt.title("Méthode à pas constante avec 4000 comme nombre d'itération")
elif k == 2:
    plt.title("Méthode à forte pente")
else:
    plt.title("Méthode à pas optimal avec la méthode de dichotomie")
points()
# # Affichage de la droite de regression linéaire.
graphique(wapprox)
#
