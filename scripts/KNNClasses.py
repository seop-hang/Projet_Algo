import json
from CalculSimilarite import CalculSimilarite

class KNNClasses:
    # L'instanciation de la classe KNNClasses prend deux arguments
    # arg1 : description(str) - la description de la classification
    # arg2 : data(list) - une liste de classes, qui correspond à la structure JSON
    def __init__(self,description,data):
        self._dsecription=description
        self._data=data
    
    def add_class(self,label:str,vectors:list):
        """
        ajouter une nouvelle classe dans data
        Input :
        arg1 : label(str) - le label de la nouvelle classe
        arg2 : vectors(list) - une liste de vecteurs de cette nouvelle classe
        """
        if len(self._data)>0:
            # on lit le data, s'il n'y a pas de cette classe dans data, on l'ajoute
            for item in self._data:
                if item['label'] == label:
                    return "La classe "+label+" existe déjà!"
            newData={"label":label,"vectors":vectors}
            self._data.append(newData)
        else:
            return "Le data est vide!"

    def add_vector(self,label:str,vector:dict):
        """
        ajouter un vecteur dans une classe qui existe déjà dans data
        Input :
        arg1 : label(str) - le label de la classe
        arg2 : vectors(dict) - le vecteur (dict) à ajouter
        """
        if len(self._data) > 0:
            # on lit le data, s'il existe déjà cette classe, on ajoute ce vecteur
            for item in self._data:
                if item["label"]==label:
                    item["vectors"].append(vector)
        else:
            return "Le data est vide!"

    def del_class(self,label:str):
        """
        supprimer la classe correspondant à label
        Input :
        arg1 : label(str) - le label de la classe que l'on veut supprimer
        """
        if len(self._data) > 0:
            # on lit le data, on supprime la classe qui correspond à label
            for item in self._data:
                if item['label']==label:
                    self._data.remove(item)
        else:
            return "Le data est vide!"

    def save_as_json(self,filename:str):
        """
        enregistrer les données d'une classe au format json
        Input :
        arg1 : filename(str) - le nom du fichier où on veut enregistrer nos données
        """
        # les données sont un dict - {'description':self._dsecription,'data':self._data}
        donnee={'description':self._dsecription,'data':self._data}
        # on essaie d'ouvrir le fichier, et enregistrer les données au format json
        try:
            with open(filename, "w") as save_file:
                json.dump(donnee, save_file)
        except Exception as err:
            return "Impossible d'ouvrir le fichier "+filename
        

    def load_as_json(self,filename:str):
        """
        charger les données depuis une classe au format json
        Input :
        arg1 : filename(str) - le nom du fichier où stockent les données
        """
        # on essaie d'ouvrir le fichier, charge les données, et les affecte aux propriétés de notre classe
        try:
            with open(filename, "r") as load_file:
                donnee = json.load(load_file)
        except Exception as err:
            return "Impossible d'ouvrir le fichier "+filename
        self._description=donnee['description']
        self._data=donnee['data']

    def classify(self,vector:dict,k:int,sim_func=CalculSimilarite.sim_cosinus):
        """
        cette fonction prend un vecteur sans label à classifier, et renvoie la liste des classes candidates
        Input :
        arg1 : vecteur(dict) - un dict contenant les tokens et leur fréquence correspondante
        arg2 : k(int) - le nombre de voisins les proches proches que l'on veut calculer
        arg3 : sim_func(func) - une fonction pour calculer la similarité entre deux vecteurs, par défaut, ce sera le
        calcule de cosinus
        Output :
        une liste des classes candidates pour le vecteur, chaque classe candidate est un dict avec les clés suivants :
        'label','n','average_sim'
        """
        distances_vect={}
        # on parcourt chaque classe dans nos données
        for eachClass in self._data:
            # on parcourt chaque vecteur dans chaque classe
            for eachVect in eachClass['vectors']:
                # on calcule la similarité entre deux vecteurs et la stocke dans distances_vect
                if eachClass['label'] in distances_vect:
                    distances_vect[eachClass['label']].append(sim_func(vector,eachVect))
                else:
                    distances_vect[eachClass['label']]=[sim_func(vector,eachVect)]
        
        # on parcout toutes les valeurs de disctionnaire distances_vect, on obtient une nouvelle liste qui contient
        # toutes les distances entre deux vecteurs
        allDists=[]
        for distances in distances_vect.values():
            for dist in distances:
                allDists.append(dist)
                
        # on ordonne cette liste en ordre décroissant et obtient une nouvelle liste qui obtient les k valeurs les plus
        # signifiantes
        if (sim_func==CalculSimilarite.sim_cosinus or sim_func==CalculSimilarite.sim_PearsonCoefficient):
            allDists.sort(reverse=True)
        else:
            allDists.sort(reverse=False)
        proches_dists = allDists[0:k]


        # on parcourt la liste des similarités les plus signifiantes, et cherche cette similarité dans distances_vect
        # pour savoir le label correspondant, ensuite, on stocke ce label dans un dict result_item comme un clé, sa valeur
        # correspondante est la similarité, si c'est la premiète ajoute, le "n" est 1, le "average_sim" est la similarité
        # telle qu'elle est, sinon, on incrémente "n", et calcule la moyenne des similarité
        result_item={}
        for dist in proches_dists:
            for eachKey in distances_vect:
                if dist in distances_vect[eachKey]:
                    if eachKey in result_item:
                        result_item[eachKey][0]=(result_item[eachKey][0]+dist)/2
                        result_item[eachKey][1]+=1
                    else:
                        result_item[eachKey]=[dist,1]

        # on reorganise le résultat
        result=[]
        for eachLabel in result_item.keys():
            result.append({'label':eachLabel,'n':result_item[eachLabel][1],'average_sim':result_item[eachLabel][0]})
            
        return result