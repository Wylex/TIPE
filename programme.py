import numpy as np
import scipy.special
import time
start_time = time.time()


class neuralNetwork:

	def __init__(self,nombreNeurones, coefficientApprentisage):
		self.nombreNeuronesCouche = nombreNeurones[:] #nombre de neurones par couche
		self.coefficientApprentisage = coefficientApprentisage
		self.fonctionActivation = lambda x: scipy.special.expit(x) #fonction d'activation, ici fonction sigmoïde

		#construction du tableau des matrices de poids
		self.poids = []
		for i in range(len(self.nombreNeuronesCouche)-1):
			self.poids.append(np.random.rand(self.nombreNeuronesCouche[i+1], self.nombreNeuronesCouche[i]) - 0.5)

	def entrainer(self, entree, cible):
		"""Fonction qui modifie les poids du réseau en fonction de l'erreur comis"""

		cible_np = np.array(cible, ndmin=2).T
		signal = np.array(entree, ndmin=2).T

		#forwardpropagation
		sortieCouches = [signal] #signaux de sortie pour chaque couche de neurones
		for i in range(len(self.nombreNeuronesCouche) -1):
			signal = np.dot(self.poids[i], signal)
			signal = self.fonctionActivation(signal)
			sortieCouches.append(signal)

		#rétropropagation
		erreurs = [cible_np - sortieCouches[-1]] #erreurs pour chaque couche de neurones
		for i in range(len(self.nombreNeuronesCouche)-2, 0,-1):
			erreurs.append(np.dot(self.poids[i].T, erreurs[-1]))

		#correction des poids en fonction des erreur de chaque neurone
		for i in range(len(self.nombreNeuronesCouche) -1):
			self.poids[i] += self.coefficientApprentisage*np.dot((sortieCouches[i+1]* erreurs[len(self.nombreNeuronesCouche)-2-i]*(1.0 - sortieCouches[i+1])), np.transpose(sortieCouches[i]))


	def calculeSortie(self, entree):
		"""Fonction qui calcule l'output"""

		signal = np.array(entree, ndmin=2).T

		for i in range(len(self.nombreNeuronesCouche) -1):
			signal = np.dot(self.poids[i], signal)
			signal = self.fonctionActivation(signal)

		return signal


#THE MNIST DATABASE OF HANDWRITTEN DIGITS
#J'ai utilisé la base de données "The MNIST Database" qui donne en un fichier csv (fichier constitué de valeurs séparées par des virgules) des exemples réels d'images de chiffres écrits à la main.
#Les images ont une taille de 28*28 pixels. Chaque image est codée dans le fichier par une ligne. Chaque ligne est alors composée d'une première valeur qui correspond au chiffre qui apparaît dans l'image (compris entre 0 et 9). La ligne est ensuite constituée de 28*28 autres valeurs comprises entre 0 et 255 qui correspondent à la couleur des pixels de l'image (en noir et blanc) permettant de reconstruire l'image.
#http://www.pjreddie.com/media/files/mnist_test.csv  (fichier contenant 60000 exemples)
#http://www.pjreddie.com/media/files/mnist_train.csv (fichier contenant 10000 exemples)


def construireEntree(ligne):
	"""À partir d'une ligne d'un fichier .csv de la base de données MNIST, construit un tableau numpy utilisable par le réseau de neurones"""

	#la ligne est initialement constituée de valeurs (de type string) séparées par des virgules
	#il faut rentrer ces valeurs dans un tableau
	valeurs = ligne.split(',')

	#ensuite on construit le tableau des entrées
	#on enlève la première valeur qui ne fait pas partie de l'image puis on converti les valeurs de type string en type float
	entree = np.array(valeurs[1:], float)

	#il reste à préparer l'entrée pour notre réseau (valeurs comprises entre 0.01 et 1)
	return (entree/255.0)*0.99 + 0.01

def construireCible(ligne):
	"""À partir d'une ligne d'un fichier .csv construit la sortie cible"""

	#la cible du réseau doit être constituée de 0 partout sauf pour le neurone correspondant au chiffre en question qui devrait valoir 1
	#les valeurs 0 et 1 posent problème car la fonction d'activation ne les atteigne jamais donc il faut s'écarter légérement de ces deux extrêmes
	cible = np.zeros(10) + 0.01
	cible[int(ligne[0])] = 0.99

	return cible

def entrainerReseau(fichierEntrainement, n):
	"""Entraîne le réseau «n» avec les exemples du fichier «fichierEntrainement»"""

	with open(fichierEntrainement) as f:
		for ligne in f: #chaque ligne du fichier correspond à un chiffre donc à un exemple d'entraînement

			entree = construireEntree(ligne)
			cible = construireCible(ligne)

			n.entrainer(entree,cible)

def determinerIndiceMax(tableau):
	"""Renvoie l'indice de l'élément maximal d'un tableau"""

	i = 0
	for j in range(1,len(tableau)):
		if(tableau[j] > tableau[i]):
			i = j

	return i





coefficientApprentissage = 0.1

n = neuralNetwork([784,300,10], coefficientApprentissage)

cheminFichierEntrainement = "mnist_train.csv"
cheminFichierTest = "mnist_test.csv"

#nombre d'itérations des exemples
for j in range(7):
	entrainerReseau(cheminFichierEntrainement, n)

#une fois le réseau  entraîné, il faut le tester
nombreExemples = 0
nombreSucces = 0

with open(cheminFichierTest) as f:
	for ligne in f:
		entree = construireEntree(ligne)
		valeurCorrecte = int(ligne[0])

		sortie = n.calculeSortie(entree)
		resultat = determinerIndiceMax(sortie)

		#vérifier la performance du réseau
		if (resultat == valeurCorrecte):
			nombreSucces += 1
		nombreExemples += 1

print("performance = ", nombreSucces/nombreExemples)
		



print("---:  %s seconds ---" % (time.time() - start_time))

