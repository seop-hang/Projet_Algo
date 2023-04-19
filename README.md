# LA CLASSIFICATION KNN
Ce projet est une implémentation de l'algorithme KNN (K-nearest neighbors) en Python pour effectuer la classification automatique de texte. Il comprend 4 fichiers de script, "TextVect.py", "KNNClasses.py", "CalculSimilarite.py" et "Main.py", un fichier de stopwords et un corpus d'apprentissage.
Voici la description de ces fichiers.

# KNNClasses.py
### Description
La classe KNNClasses permet de créer une instance de classification de KNN en fournissant une description et une liste de classes qui correspondent à la structure JSON. Elle dispose de méthodes pour ajouter une nouvelle classe, ajouter un vecteur dans une classe existante, supprimer une classe, enregistrer et charger les données au format JSON, et classifier un vecteur.

### Prérequis
La classe KNNClasses nécessite l'installation préalable du module json et du module TextVect. Si il n'y a pas d'autre méthodes à calculer la similarité entre deux vecteurs, la méthode sim_cosinus du module TextVect est alors utilisé pour le calcul.

### Création de l'instance de la classe KNNClasses
L'instanciation de la classe KNNClasses prend deux arguments :
- description : description de la classification (str).
- data : liste de classes, qui correspond à la structure JSON (list).

### API
### add_class(label, vectors)
Cette méthode permet d'ajouter une nouvelle classe dans la liste data.
- label : label de la nouvelle classe (str).
- vectors : liste de vecteurs de la nouvelle classe (list).

### add_vector(label, vector)
Cette méthode permet d'ajouter un vecteur dans une classe existante dans la liste data.
- label : label de la classe (str).
- vector : vecteur (dict) à ajouter.

### del_class(label)
Cette méthode permet de supprimer la classe correspondant à label.
- label : label de la classe à supprimer (str).

### save_as_json(filename)
Cette méthode permet d'enregistrer les données d'une instance de la classe KNNClasses au format JSON.
- filename : nom du fichier où les données doivent être enregistrées (str).

### load_as_json(filename)
Cette méthode permet de charger les données d'une instance de la classe KNNClasses depuis un fichier au format JSON.
- filename : nom du fichier où les données sont stockées (str).

### classify(vector, k, sim_func)
Cette méthode permet de classifier un vecteur et de renvoyer la liste des classes candidates.
- vector : vecteur (dict) à classifier.
- k : nombre de voisins les plus proches que l'on veut calculer (int).
- sim_func : fonction pour calculer la similarité entre deux vecteurs (par défaut, CalculSimilarite.sim_cosinus; et d'autres méthodes au choix sont : CalculSimilarite.sim_PearsonCoefficient, CalculSimilarite.sim_EucDist, CalculSimilarite.sim_ManDist).


# TextVect.py
### Description
La classe TextVect fournit une implémentation de base pour la vectorisation de texte. Elle permet de créer une instance de corpus d'apprentissage en fournissant le nom du répertoire où se stockent les fichiers, et elle dispose de méthodes pour lire le fichiers de stopwords, tokeniser le texte, vectoriser le textes, filtrer les vecteurs, calculer le TF.IDF et calculer le cosinus.

### Prérequis
La classe KNNClasses nécessite l'installation préalable des modules suivants : re, copy, math, et os.

### API
### read_dict(stoplist_filename)
La méthode read_dict prend un nom de fichier en entrée et renvoie une liste de stopwords.
- stoplist_filename: (str) - nom du fichier contenant les stopwords. Un mot par ligne.

### tokenize(self, text)
La méthode tokenize prend un texte en entrée et renvoie une liste de tokens. La méthode utilise une expression régulière pour découper le texte en tokens.
- text: str - le texte à tokeniser.

### vectorise(self, tokens)
La méthode vectorise prend une liste de tokens en entrée et renvoie un dictionnaire contenant le vocabulaire avec les fréquences associées.
- tokens: list(str) - la liste des tokens.

### doc2vec(self, data_path, dir_or_filename)
La méthode doc2vec prend en entrée un nom de fichier ou un dossier et renvoie un dictionnaire contenant un label et une liste de vecteurs.
- data_path: str - le répertoire où sont stockés tous les fichiers.
- dir_or_filename: str - le nom du fichier ou du dossier à lire.

### filtrage(self, stoplist, documents, non_hapax)
La méthode élimine tous les vocables appartenant à la stoplist à partir d'une liste de documents (objets avec deux propriétés 'label' et 'vectors')
- stoplist : set - l'ensemble des stopwords
- documents : list(doc) - un doc est un dict contenant deux clés : 'label' et 'vectors' et doc : { 'label':str, 'vectors':dict }
- non_hapax : bool - indique si on veut éliminer les hapax (True) ou non (False)

### tf_idf(self, documents: list)
cette fonction calcule le TF-IDF pour une liste de documents et revoie une liste de dictionnaires, chaque dictionnaire contient une modification des fréquences associées à chaque mot. Les fréquences ont été divisées par le logarithme de la fréquence de documents.
-documents : une liste de dictionnaires contenant des informations sur les documents (label, vectors).

### generate_data(self, data_path='./corpus', non_hapax=True):
cette fonction génère une base de données à partir de fichiers textes et renvoie une liste de dictionnaires, chaque dictionnaire contient des informations sur les documents (label, vectors). Les vecteurs sont générés en utilisant la méthode doc2vec.
- data_path : le chemin du dossier contenant les fichiers à utiliser pour générer la base de données
- non_hapax : un booléen qui indique si nous voulons éliminer les hapax ou non.

### generate_vect(self, filename, data_path='./corpus', non_hapax=True):
cette fonction génère un vecteur pour un fichier texte donné, ce vecteur est un dict contenant les mots et leur fréquence TF-IDF. 
- filename : le nom du fichier à vectoriser
- data_path : le chemin du dossier contenant les fichiers à utiliser pour générer la base de données
- non_hapax : un booléen qui indique si nous voulons éliminer les hapax ou non.

# CalculSimilarite.py
### Description
La classe CalculSimilarite regroupe 4 fonctions pour caclculer la similarité entre deux vecteurs, "sim_cosinus", "sim_PearsonCoefficient", "sim_EucDist" et "sim_ManDist" qui calculent respectivement le cosinus, le coefficient de corrélation de Pearson, la distance euclidienne et la distance de manhattan entre deux vecteurs. Parmi eux, les deux premières fonctions retournent une valeur entre -1 et 1, et une valeur plus proche de 1 signifie une similarité plus élevée. Les deux autres retournent une valeur de distance, une distance plus faible signifie une similarité plus élevée.

### Prérequis
La classe KNNClasses nécessite l'installation préalable des modules suivants : math, numpy.

### API
### scalaire(vector1, vector2):
cette fonction calcule le produit scalaire de deux vecteurs.
- arg1 : vector1 - hash
- arg2 : vector2 - hash

### norme(vector):
cette fonction calcule la norme d'un vecteur.
- arg1 : vector - hash

### sim_cosinus(cls, vector1, vector2):
cette fonction calcule la mesure de similarité cosinus entre deux vecteurs.
- arg1 : vector1 - hash
- arg2 : vector2 - hash

### sim_PearsonCoefficient(cls,vector1,vector2)
Cette fonction récupère deux vecteurs sous forme de hashage, et renvoie le coefficient de corrélation de Pearson
- arg1 : vector1 - hash
- arg2 : vector2 - hash

### sim_EucDist(cls,vector1,vector2)
- arg1 : vector1 - hash
- arg2 : vector2 - hash

### sim_ManDist(cls,vector1,vector2)
- arg1 : vector1 - hash
- arg2 : vector2 - hash

# Main.py
### Description
C'est le fichier d'entrée, on utilise les deux classes "TextVect" et "KNNClasses" dans ce fichier afin de faire la classification.

### Prérequis
Pour exécuter ce fichier, il faut l'installation préalable des modules "TextVect" et "KNNClasses".

# La base de données
Pour construire la base de données, il faut stocker les fichiers de différentes classes dans un même dossier, par exemple, si on a 5 recettes de l'entrée, 5 recettes du plat et 5 recettes du dessert, on les stocke respectivement dans le dossier qui s'appelle "entrée","plat" et "dessert". Et ces dossiers sont stocker dans un même dossier, par défaut, on le donne le nom "corpus".
Quand on a un fichier dont on ne connaît pas la classe, on stocke ce fichier dans le dossier "corpus".

# Utilisation
Dans le fichier "Main.py", on écrit les scripts sous la fonction "__main__".
### L'utilisation de base
Afin de faire la classification, il faut un corpus d'apprentissage, alors, on instancier la classe TextVect et obtient une instance "corpus".
```corpus=TextVect()```
ensuite, on utilise la méthode generate_data de l'instance "corpus", pour le faire, il faut donner le nom du répertoire, par défaut, ce sera "../corpus".
```data = corpus.generate_data()```
Afin de classifier un texte, on utilise la méthode generate_vect de l'instance "corpus", pour le faire, il faut donner le nom du répertoire, par défaut, ce sera "../corpus", et il faut aussi donner le nom du fichier à classifier, il faut vérifier que ce fichier est dans le répertoire.
```vector=corpus.generate_vect('dessert_classfier.txt')```
Après le travail de la préparation, on instancier la classe "KNNClasses" en donnant une description et notre data comme arguments.
```KNNClass=KNNClasses('La classification des recettes',data)```
ensuite, on utilise la méthode "classify" de l'instance "KNNClass" en donnant notre vecteur à classifier et un nombre "k" (nombre de plus proches voisins) comme arguments.
```res=KNNClass.classify(vector,3)```
on obtient le résultat et on l'affiche
```print(res)```

### L'utilisation complémentaire
Pour ajouter une classe dans notre instance "KNNClass":
```KNNClass.add_class(label,vectors)```
Pour ajouter un vecteur dans une classe:
```KNNClass.add_vector(label,vector)```
Pour supprimer une classe dans notre instance "KNNClass":
```KNNClass.del_class(label)```
Pour enregistrer les données d'une classe au format json:
```KNNClass.save_as_json(filename)```
Pour charger les données depuis une classe au format json:
```KNNClass.load_as_json(filename)```

# Améliorations Possibles
- Lorsque l'on génère les vecteurs de base de données, et losque l'on ajoute une nouvelle classe, on n'a pas vérifié si il existe des vecteurs qui sont identiques, autrement dit, s'il y a des vecteurs qui sont les même, la distance entre les vecteurs seront identique, qui entraînera des problèmes. Donc, il est nécessaire d'ajouter des vérifications.
- Quant à la structure de data, on n'a stocké qu'une liste de vecteurs, mais il n'y a pas de données supplémentaires liées à chaque vecteur, à titre d'exemple, le nom de chaque document, etc. Et il est possible d'ajouter ces informations quand on génère des vecteurs.
- Quand on compare la similarité entre deux vecteurs, le calcule de distance est toujours influencé par le longeur de document. Plus précisément, même si les deux documents sont de même classe, la distance pourrait être plus grande si l'un de ces documents est excessivement long. Par conséquent, il faut des améliorations pour traiter les documents avec une longueur très différente.