import math
import numpy as np

class CalculSimilarite():
    """
    les fonctions pour calculer la similarité cosinus
    """
    @staticmethod
    def scalaire(vector1,vector2):
        """
        Cette fonction récupère deux vecteurs sous forme de hashage
        et renvoie leur produit scalaire
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : un produit scalaire - float
        """
        liste_scalaire=[]
        for key in vector1:
            if key in vector2:
                liste_scalaire.append(vector1[key]*vector2[key])
        produit_scalaire=sum(liste_scalaire)
        return produit_scalaire

    @staticmethod
    def norme(vector):
        """
        Cette fonction récupère un vecteur sous forme de hashage
        et renvoie sa norme
        Input :
            arg1 : vector - hash
        Output :
            valeur de retour : une norme - float
        """
        norme_carre=0
        for key in vector:
            norme_carre+=vector[key]*vector[key]
        norme=math.sqrt(norme_carre)
        return norme

    @classmethod
    def sim_cosinus(cls,vector1,vector2):
        """
        Cette fonction récupère deux vecteurs sous forme de hashage,
        et renvoie leur cosinus
        en appelant les fonctions Scalaire et Norme
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : un cosinus - float
        """
        norme1=cls.norme(vector1)
        norme2=cls.norme(vector2)
        scal=cls.scalaire(vector1,vector2)
        cosinus=(scal/(norme1*norme2))
        return cosinus

    @classmethod
    def sim_PearsonCoefficient(cls,vector1,vector2):
        """
        Cette fonction récupère deux vecteurs sous forme de hashage,
        et renvoie le coefficient de corrélation de Pearson
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : le coefficient de corrélation de Pearson - float
        """
        # récupérer les mots communs dans les deux hash
        mots_commun = set()
        for mot in vector1:
            if mot in vector2:
                mots_commun.add(mot)
        # rendre les deux vecteurs à la même dimension
        for word in vector1:
            if word not in mots_commun:
                vector2[word] = 0
        for word in vector2:
            if word not in mots_commun:
                vector1[word] = 0
        # calculer la distance manhattan
        vect1 = np.array([val for val in vector1.values()])
        vect2 = np.array([val for val in vector2.values()])
        # calculer la moyenne des vecteurs
        vect1_mean = np.mean(vect1)
        vect2_mean = np.mean(vect2)
        # calculer la variance coefficient
        covariance = np.sum((vect1 - vect1_mean) * (vect2 - vect2_mean))
        # calculer les écarts standards
        vect1_std = np.sqrt(np.sum((vect1 - vect1_mean) ** 2))
        vect2_std = np.sqrt(np.sum((vect2 - vect2_mean) ** 2))
        # calculer le coefficient de corrélation de Pearson
        coefficient = covariance / (vect1_std * vect2_std)
        return coefficient

    @classmethod
    def sim_EucDist(cls,vector1,vector2):
        """
        Cette fonction récupère deux vecteurs sous forme de hashage,
        et renvoie leur distance euclidienne
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : une distance euclidienne - float
        """
        vector=[]
        # rendre les deux vecteurs à la même dimension
        for vect in vector1:
            if vect not in vector2:
                vector.append(vector1[vect]-0)
            else:
                vector.append(vector1[vect]-vector2[vect])
        for vect in vector2:
            if vect not in vector1:
                vector.append(vector2[vect]-0)
            else:
                vector.append(vector2[vect]-vector1[vect])
        # calculer la distance euclidienne
        dist=np.linalg.norm(vector)
        return dist

    @classmethod
    def sim_ManDist(cls,vector1,vector2):
        """
        Cette fonction récupère deux vecteurs sous forme de hashage,
        et renvoie leur distance de manhattan
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : une distance de manhattan - float
        """
        # récupérer les mots communs dans les deux hash
        mots_commun=set()
        for mot in vector1:
            if mot in vector2:
                mots_commun.add(mot)
        # rendre les deux vecteurs à la même dimension
        for word in vector1:
            if word not in mots_commun:
                vector2[word]=0
        for word in vector2:
            if word not in mots_commun:
                vector1[word]=0
        # calculer la distance manhattan
        vect1=np.array([val for val in vector1.values()])
        vect2=np.array([val for val in vector2.values()])
        dist=np.sum(np.abs(vect1-vect2))
        return dist