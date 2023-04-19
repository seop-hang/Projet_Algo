from TextVect import TextVect
from KNNClasses import KNNClasses
from CalculSimilarite import CalculSimilarite

if __name__=="__main__":
    # Afin de faire la classification, il faut un corpus d'apprentissage, alors, on instancie la classe TextVect et
    # obtient une instance "corpus"
    corpus=TextVect()
    # ensuite, on utilise la méthode generate_data de l'instance "corpus", pour le faire, il faut donner le nom du
    # répertoire, par défaut, ce sera "../corpus"
    data = corpus.generate_data()
    # ensuite, on utilise la méthode generate_vect de l'instance "corpus", pour le faire, il faut donner le nom du
    # répertoire, par défaut, ce sera "../corpus", et il faut aussi donner le nom du fichier à classifier, il faut
    # vérifier que ce fichier est dans le répertoire
    vector=corpus.generate_vect('dessert_classfier.txt')

    # Après le travail de la préparation, on instancier la classe "KNNClasses" en donnant une description et notre data
    # comme arguments
    KNNClass=KNNClasses('La classification des recettes',data)

    # on peut tester la fonction add_class, pour le faire, on stocke les fichiers de la nouvelle classe dans le répertoire
    # "../newclass/boisson", et génère des vecteurs avec l'instance "corpus", le résultat est une liste, on prend le
    # premier élément et récupère les vecteurs
    newClass=corpus.generate_data("../newclass")[0]['vectors']
    KNNClass.add_class("boisson",newClass)

    # on peut tester la fonction add_vector, pour le faire, on stocke un fichier dans le répertoire "../newclass", et
    # génère un vecteur avec l'instance "corpus", on ajoute ce vecteur avec la fonction "add_vector"
    newVector=corpus.generate_vect('../newclass/boisson_test.txt')
    KNNClass.add_vector("boisson",newVector)

    # le test de la fonction "del_class"
    KNNClass.del_class("boisson")

    # On essaie la fonction save_as_json
    KNNClass.save_as_json("../data/data.json")

    # ensuite, on utilise la méthode "classify" de l'instance "KNNClass" en donnant notre vecteur à classifier, un
    # nombre "k" (nombre de plus proches voisins) comme arguments, et une méthode à calculer la similarité, par défaut,
    # ce sera sim_cosinus
    res=KNNClass.classify(vector,3,CalculSimilarite.sim_ManDist)
    # on obtient le résultat et on l'affiche
    print(res)