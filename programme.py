import numpy as np
import matplotlib.pyplot
import scipy.special
import time
start_time = time.time()

class neuralNetwork:

	def __init__(self,nombreNeurones, coefficientApprentisage):
		self.nombreNeurones = nombreNeurones[:] #tableau qui donne le nombre de neurones par couche
		self.coefficientApprentisage = coefficientApprentisage

		self.fonctionActivation = lambda x: scipy.special.expit(x) #Fonction d'activation

		#Construction du tableau des matrices de poids
		self.poids = []
		for i in range(len(nombreNeurones)-1):
			self.poids.append(np.random.rand(self.nombreNeurones[i+1], self.nombreNeurones[i]) - 0.5)

	#deux phases: calculer l'output, rétropropagation des erreurs pour modifier poids
	def entrainer(self, entree, cible):
		cible_np = np.array(cible, ndmin=2).T
		signal = np.array(entree, ndmin=2).T

		#Forwardpropagation
		sortieCouches = [signal]
		for i in range(len(self.nombreNeurones) -1):
			signal = np.dot(self.poids[i], signal)
			signal = self.fonctionActivation(signal)
			sortieCouches.append(signal)

		#Backpropagation
		erreurs = [cible_np - sortieCouches[-1]]
		for i in range(len(self.nombreNeurones)-2, 0,-1):
			erreurs.append(np.dot(self.poids[i].T, erreurs[-1]))

		#Corrige les poids en fonction des erreur de chaque neurone
		for i in range(len(self.nombreNeurones) -1):
			self.poids[i] += self.coefficientApprentisage*np.dot((sortieCouches[i+1]* erreurs[len(self.nombreNeurones)-2-i]*(1.0 - sortieCouches[i+1])), np.transpose(sortieCouches[i]))

	def calculeSortie(self, entree):
		signal = np.array(entree, ndmin=2).T

		for i in range(len(self.nombreNeurones) -1):
			signal = np.dot(self.poids[i], signal)
			signal = self.fonctionActivation(signal)

		return signal

#Valeurs initiales pour les poids doivent être aléatoires et petites

#THE MNIST DATABASE OF HANDWRITTEN DIGITS
#J'ai utilisé la base de données "The MNIST Database" qui donne en un fichier csv, c'est à dire avec des valeurs séparées par des virgules, des exemples réels d'images de chiffres écrits à la main.
#Chaque chiffre écrit à la main est transformé en une image de taille 28*28 pixels. Chaque image est codé dans le fichier par une ligne. Chaque ligne est alors composé d'une première valeur qui est un chiffre entre 0 et 9 et qui correspond au chiffre qui apparaît dans l'image et qu'on souhaite que le réseau soit capable de reconnaître. La ligne est constitué ensuite de 28*28 autres valeurs comprises entre 0 et 255 qui correspondent à la couleur des pixels de l'image (en noir et blanc) et qui permettraient de reconstruire l'image.
#http://www.pjreddie.com/media/files/mnist_test.csv  (fichier contenant 60000 exemples)
#http://www.pjreddie.com/media/files/mnist_train.csv (fichier contenant 10000 exemples)


#À partir d'une ligne d'un fichier csv construit un tableau numpy utilisable par le réseau de neurones
def construireEntree(ligne):
	#La ligne est initialement constitué de valeurs (de type string) séparées par des virgules
	#Il faut donc isoler les valeurs dans un tableau
	valeurs = ligne.split(',')

	#Il faut ensuite construire le tableau des entrées.
	#On enlève la première valeur qui ne fait pas partie de l'image puis en converti les valeurs de type string en type float
	entree = np.array(valeurs[1:], float)

	#Il reste à préparer l'entrée pour notre réseau
	return (entree/255.0)*0.99 + 0.01

#À partir d'une ligne d'un fichier csv construit la sortie théorique du réseau
def construireCible(ligne):
	#La cible du réseau doit être constitué de 0 partout sauf pour le neurone correspondant au chiffre en question qui devraie valoir 1.
	#Les valeurs 0 et 1 posent problème qui la fonction d'activation ne les atteigne jamais donc on s'écarte légérement de ces 2 extrêmes
	cible = np.zeros(output_nodes) + 0.01
	cible[int(ligne[0])] = 0.99

	return cible

def entrainerReseau(fichierEntrainement, n):
	with open(fichierEntrainement) as f:
		for ligne in f: #Chaque ligne du fichier correspond à un chiffre donc à un exemple d'entraînement

			entree = construireEntree(ligne)
			cible = construireCible(ligne)

			n.entrainer(entree,cible)

#Renvoie l'indice de l'élément maximal d'un tableau
def determinerIndiceMax(tableau):
	i = 0
	for j in range(1,len(tableau)):
		if(tableau[j] > tableau[i]):
			i = j

	return i




input_nodes = 784
hidden_nodes = 20
output_nodes = 10

learning_rate = 0.1

#n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#n = neuralNetwork([input_nodes,hidden_nodes,output_nodes], learning_rate)

tableau = []
for nodes in range(320, 500, 30):
	performance = 0
	for i in range(50):
		n = neuralNetwork([784,nodes,10],learning_rate)

		cheminFichierEntrainement = "mnist_dataset/mnist_train.csv"
		cheminFichierTest = "mnist_dataset/mnist_test.csv"

		entrainerReseau(cheminFichierEntrainement, n)

		#Une fois le réseau  entraîné, il faut le tester
		nombreExemples = 0
		nombreSucces = 0

		with open(cheminFichierTest) as f:
			for ligne in f:
				entree = construireEntree(ligne)
				valeurCorrecte = int(ligne[0])

				sortie = n.calculeSortie(entree)
				resultat = determinerIndiceMax(sortie)

				#Vérifier la performance du réseau
				if (resultat == valeurCorrecte):
					nombreSucces += 1
				nombreExemples += 1

		#print("performance = ", nombreSucces/nombreExemples)
		performance += nombreSucces/nombreExemples

	print("nodes = ", nodes, " performance = ", performance/50.0)
	tableau.append((nodes, performance/50.0))

print(tableau)


print("---:  %s seconds ---" % (time.time() - start_time))

