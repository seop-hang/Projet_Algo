import re
import copy
import math
import os

class TextVect():
    def __init__(self):
        self.tok_grm=re.compile(r"""(?:etc.|p.ex.|cf.|M.)|\w+(?=(?:-(?:je|tu|ils?|elles?|nous|vous|leur|lui|les?|ce|t-|même|ci|là)))|[\w\-]+'?|.""",re.X)
        self.freq_doc={}

    @staticmethod
    def read_dict(stoplist_filename):
        """
        Lecture d'une stoplist à partir d'un fichier
        Input : 
        arg1 : str - nom du fichier à lire. Un mot par ligne.
        Output :
        valeur de retour : set(str) - ensemble des stopwords
        """
        try:
            # on ouvre, lit et après ferme le fichier
            dict_file = open(stoplist_filename, "r", encoding="utf-8")
            dict_content = dict_file.read()
            dict_file.close()
            # on sépare le dict_content(string) avec la saut de ligne et renvoie une liste
            stoplist = set(dict_content.split("\n"))
        except Exception as err:
            return "Impossible d'ouvrir le fichier de stopwords"
        return stoplist

    def tokenize(self,text):
        """
        Input : 
            arg1 : un texte à tokeniser
        Output : 
            valeur de retour : la liste des tokens
        """
        tokens=self.tok_grm.findall(text)
        return tokens

    def vectorise(self,tokens):
        """
        But : cette fonction récupère une liste de tokens, et renvoie un dictionnaire
        contenant le vocabulaire avec les fréquences associées
        Input :
            arg1 : tokens - list(str)
        Output :
            valeur de retour : un dict contenant les vocables (clés) et les fréq associées (valeur)
        """
        token_freq={}  # initialisation du hachage
        for token in tokens:  # parcours des tokens
            if token!=" ": #on inignore les token vides (espace)
                if token not in token_freq.keys():   # si on a un token
                    token_freq[token]=1
                token_freq[token]+=1 #on associe la fréquence à la clé (token)
        return token_freq
    
    def doc2vec(self,data_path,dir_or_filename):
        """
        But : cette fonction récupère un nom d'un fichier ou un dossier, et renvoie un hashage transformant le contenu
        des fichiers en vecteurs
        Input :
            arg1 : data_path(str): le répertoire où stockent tous les fichiers
            arg2 : dir_or_filename - str
        Output :
            valeur de retour : un dict contenant un label et une liste de vecteurs
        """
        path=data_path+'/'+dir_or_filename
        # si c'est un fichier, on le lit, tokenise et vectorize le contenu
        if os.path.isfile(path):
            try:
                input_file=open(data_path+'/'+dir_or_filename,mode="r",encoding="utf8")
            except Exception as err:
                return "Impossible d'ouvrir le fichier"+dir_or_filename
            tokens=[]
            for line in input_file:
                line=line.strip() #on supprime les retours à la ligne
                toks=self.tokenize(line)
                tokens.extend(toks)
            input_file.close()
            # vector=self.vectorise(tokens)
            vector = {'label': dir_or_filename, 'vectors': [self.vectorise(tokens)]}
        # si c'est un dossier, on parcourt ce dossier, on lit, tokenise et vectorize chaque fichier dans ce dossier
        if os.path.isdir(path):
            filenames = [file.name for file in os.scandir(data_path+'/' + dir_or_filename) if file.is_file()]
            liste_vectors = []
            for file_name in filenames:
                try:
                    input_file = open(data_path+'/'+dir_or_filename+'/'+file_name, mode="r", encoding="utf8")
                except Exception as err:
                    return "Impossible d'ouvrir le fichier" + file_name
                tokens = []
                for line in input_file:
                    line = line.strip()
                    toks = self.tokenize(line)
                    tokens.extend(toks)
                input_file.close()
                liste_vectors.append(self.vectorise(tokens))
            vector={'label': dir_or_filename, 'vectors': liste_vectors}
        return vector

    def filtrage(self,stoplist,documents,non_hapax):
        """
        A partir d'une liste de documents (objets avec deux propriétés 'label' et 'vectors')
        on élimine tous les vocables appartenant à la stoplist.
        Input :
          arg1 : set - l'ensemble des stopwords
          arg2 : list(doc) - un doc est un dict contenant deux clés : 'label' et 'vectors'
                doc : { 'label':str, 'vectors':dict }
          arg3 : bool - indique si on veut éliminer les hapax (True) ou non (False)
        """
        # on l'initialise une liste à stocker tous les docs filtrés
        documents_filtre=[]
        for document in documents:
            document_filtre={'label':document['label'],'vectors':[]}
            vectors=document['vectors']
            for vector in vectors:
                vector_filtre = {}
                # on parcourt chaque token dans les clés de dict vectors
                for token in vector.keys():
                # selon le choix de l'utilisateur, si on veut éliminer les hapax, on exécute les codes ci-desous
                    if token.lower() not in stoplist and (not non_hapax or vector[token]>1):
                        vector_filtre[token]=vector[token]
                # Après avoir terminé l'ajout des tokens dans le dictionnaire des tokens filtrés, on l'initialise
                # dans le dictionnaire document_filtre
                document_filtre['vectors'].append(vector_filtre)
            documents_filtre.append(document_filtre)
        return documents_filtre

    def tf_idf (self,documents:list)->list:
        """
        Calcul du TF.IDF pour une liste de documents
        Input :
          arg1 : list(dict) : une liste de documents ...
        Output :
          valeur de retour : une liste de documents avec une modification des fréq
          associées à chaque mot (on divise par le log de la fréq de documents)
        """
        #création d'un dict contenant tous les mots de tous les docs
        mots=set()

        # 1. on crée l'ensemble de tous les mots
        # on parcours les documents
        for doc in documents:
            for vector in doc['vectors']:
            #pour chaque mot du doc étant dans notre vecteur doc
            #word = notre variable qui récupère chaque mot
                for word in vector:
                    mots.add(word)

        # 2. on parcourt tous les mots pour calculer la fréquence de doc de chacun
        for word in mots:
            # on parcourt les documents
            for doc in documents:
                for vector in doc['vectors']:
                    if word in vector:
                        if word not in self.freq_doc:
                            self.freq_doc[word]=1
                        else :
                            self.freq_doc[word]+=1

        # 3. on parcourt les docs mot par mot pour mettre à jour la fréquence
        documents_new=copy.deepcopy(documents)
        for doc in documents_new:
            for vector in doc['vectors']:
                for word in vector:
                    vector[word] = vector[word] / math.log(1 + self.freq_doc[word])
        return documents_new

    # le calcule de tfidf d'un seul vecteur
    def tfidf_vect(self,vector):
        """
        Calcul du TF.IDF pour un seul vecteur
        Input :
          arg1 : le vecteur à calculer
        Output :
          valeur de retour : une liste de documents avec une modification des fréq
          associées à chaque mot (on divise par le log de la fréq de documents)
        """
        for word in vector:
            if word in self.freq_doc.keys():
                self.freq_doc[word]+=1
            else:
                self.freq_doc[word]=1
            vector[word] = vector[word] / math.log(1 + self.freq_doc[word])
        return vector

    def generate_data(self,data_path='../corpus',non_hapax=True):
        """
        BUT: Cette fonction génère une base de données, elle prend le nom du répertoire, et dans ce répertoire, il y a
        plusieurs dossiers, chaque dossier contient des fichiers d'une même classe. Cette fonction parcourt ces dossiers,
        et génère des dists, chaque dist : {‘label':nom du dossier,'vectors':une liste des vecteurs transformés de fichiers
        de chaque dossier
        Input :
          arg1 : data_path(str): le répertoire où stockent tous les fichiers, par défaut, ce sera '../corpus'
          arg2 : non_hapax (bool) : indique si on veut éliminer les hapax (True) ou non (False), par défaut, ce sera True
        Output :
          valeur de retour : une liste de documents, chaque document : {'label': nom du dossier,'vectors':tf_idf}

        """
        dirnames=[]
        for eachDir in os.scandir(data_path):
            if eachDir.is_dir():
                dirnames.append(eachDir.name)
        data=[]
        for dirname in dirnames:
            vector=self.doc2vec(data_path,dirname)
            data.append(vector)
        stoplist=self.read_dict('../stopwords.txt')
        data_filtree=self.filtrage(stoplist,data,non_hapax)
        data_tfidf=self.tf_idf(data_filtree)
        return data_tfidf

    def generate_vect(self,filename,data_path='../corpus',non_hapax=True):
        """
        BUT: Cette fonction génère un vecteur à partir d'un texte à vectoriser, elle est utilisée pour vectoriser le fichier
        que nous voulons classifier
        Input :
          arg1 : other (une autre instance de la classe "TextVect"
          arg2 : filename (str) : le nom du fichier à vectoriser
          arg3 : data_path(str): le répertoire où stockent tous les fichiers, par défaut, ce sera './corpus'
          arg4 : non_hapax (bool) : indique si on veut éliminer les hapax (True) ou non (False)
        Output :
          valeur de retour : une dict contenant des mots et sa fréquence tf_idf
        """
        data = []
        vector=self.doc2vec(data_path,filename)
        data.append(vector)
        stoplist = self.read_dict('../stopwords.txt')
        data_filtree = self.filtrage(stoplist, data, non_hapax)
        # on obtient le vecteur de ce fichier à classifier
        vect=data_filtree[0]["vectors"][0]
        # on utilise la fonction tfidf_vect en envoyant ce vecteur, et la propriété (freq_doc) d'une autre instance (le
        # corpus d'apprentissage) comme arguments
        vect_tfidf = self.tfidf_vect(vect)
        return vect_tfidf