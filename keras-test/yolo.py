import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K

from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, body_yolo, tiny_body_yolo
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _base = {                                           #création d'un dico pour associer les différents éléments
        "model_path": 'model_data/yolo.h5',                 #chemin pour le model
        "anchors_path": 'model_data/tiny_yolo_anchors.txt', #chemin pour les anchors
        "classes_path": 'model_data/coco_classes.txt',      #chemin pour les classes
        "score": 0.3,                                       #donne une valeur de base au score
        "iou": 0.45,
        "model_image_size": (416, 416),                     #donne la dimention de l'image
        "gpu_num": 1,                                       #donne un numéro au gpu

    }
    @classmethod
    def get_base(cls, n):
        if n in cls._base:
            return cls._base[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self,**kwargs): #le constructeur
        #La méthode update () met à jour le dictionnaire avec les éléments d'un autre  dictionnaire
        self.__dict__.update(self._base) #mise à jour avec les éléments de defauts
        self.__dict__.update(kwargs)
        self.name_class = self._getclass() #obtention des noms des classes grâce à la fonction _getclass()
        self.anchors = self._getanchors()  #obtention des données des anchors grâce à la fonction _getanchors()
        self.sess = K.get_session()        #création d'un session avec Keras
        self.boxs, self.scores, self.classes = self.creation() #permet d'avoir les boxs, les scores et classes grâce à la fonction création()

    # permet de lire le fichier avec les anchors et d'enregistrer les valeurs

    def _getanchors(self):
        anchors_path = os.path.expanduser(self.anchors_path) # enregistrement de la valeur du nom d'anchors
        with open(anchors_path) as f:                        # ouverture du fichier
            anchors = f.readline()                           # lecture de la ligne
        anchors_vec = [float(j) for j in anchors.split(',')] # sépare les valeurs à chaque virgule et les transforme en float
        return np.array(anchors_vec).reshape(-1,2)           # redimensionne pour associer chaque duo de valeur

    def creation(self):
        model_path = os.path.expanduser(self.model_path) #lecture du nom du fichier
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.' #vérifier que fichier fini bien par .h5

        anchors_taille = len(self.anchors) # le nompbre d'anchors
        classes_taile = len(self.name_class) # le nombre de classe
        version_tiny = anchors_taille == 6    #lorsque c'est la version tiny ==> le nombre d'anchors est égale à 6

        try:
            self.model_yolo = load_model(model_path, compile=False) #chragement du model grâce à la fonction de Keras
        except:
            self.model_yolo = tiny_yolo_body(Input(shape=(None,None,3)), anchors_taille//2, classes_taile) \
                if version_tiny else  yolo_body(Input(shape=(None,None,3)),anchors_taille//3,classes_taile)
            self.model_yolo.load_weights(self.model_path)  # s'assurer que le modèle, les ancres et les classes correspondent

        else:
            assert self.model_yolo.layers[-1].output_shape[-1] == \
                anchors_taille/len(self.model_yolo.output) * (classes_taile +5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, et classes sont chargé.'.format(model_path)) #afficher que le model, les ancors et les classes sont bien importé

        # Génère des couleurs pour dessiner des cadres de délimitation.
        hsv_tuples = [(x / len(self.name_class), 1., 1.)for x in range(len(self.name_class))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list( map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),self.colors))

        np.random.seed(10101)  # Correction des couleurs cohérentes entre les courses.
        np.random.shuffle(self.colors)  # Mélangez les couleurs pour décorréler les classes adjacentes.
        np.random.seed(None)  # Réinitialiser la valeur par défaut.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,)) # création d'un tenseur  pour filtrer les boxes

        if self.gpu_num >= 2: #si il y a un gpu
            self.yolo_model = multi_gpu_model(self.model_yolo, gpus=self.gpu_num) #utilisation du gpu pour le model yolo
        boxes, scores, classes = yolo_eval(self.model_yolo.output, self.anchors,len(self.name_class), self.input_image_shape,score_threshold=self.score, iou_threshold=self.iou) #utilisation de la fonction eval

        return boxes, scores, classes

    def image_detection(self, img):
        start =timer() #prends la valeur du timer
        topvec, leftvec, bottomvec, rightvec = [], [], [], []
        if self.model_image_size != (None,None):  #si la taille d'image n'est pas nul
            assert self.model_image_size[0]%32 == 0, 'Multiple de 32 nécessaire'   #vérifie que la taille de l'image est bien un multiple de 32
            assert self.model_image_size[1] % 32 == 0, 'Multiple de 32 nécessaire'
            box = letterbox_image(img, tuple(reversed(self.model_image_size))) #utilisation de la fonction letterbox_image de yolo3.utils pour avoir la box


        else: #si la taille de l'image est nulle
            new_img =(img.width - (img.width % 32),img.height - (img.heaight % 32)) #création de la nouvelle image
            box = letterbox_image(img, new_img) #utilisation de la fonction letterbox_image de yolo3.utils pour avoir la box
        img_data = np.array(box, dtype='float32')  #création d'un vecteur avec les données de box

        print(img_data.shape)
        img_data /= 255.
        img_data = np.expand_dims(img_data, 0)

        #permet d'obtenir les différentes boxes , lse score et les classe preditent  des objets
        out_boxes, out_scores, out_classes = self.sess.run( [self.boxs, self.scores, self.classes],
            feed_dict={self.model_yolo.input: img_data,self.input_image_shape: [img.size[1], img.size[0]],K.learning_phase(): 0}) #TODO : errreur ici

        print('{} boxes trouver '.format(len(out_boxes))) #print le nombre de boxe trouver

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32')) #permet de définir la police d'écriture et la taille

        thick = (img.size[0] + img.size[1]) //300 #on prends la somme de la  hauteur et de la largeur qu'on divise par 300

        for i, p in reversed(list(enumerate(out_classes))):
            #enumerate permet d'associer un numéro à chaque valeur de out_classes (i) et p est la valeur
            classes_predict = self.name_class[p] #enrgistre le nim de la classe par rapport à la valeur p
            box = out_boxes[i] #enregistre les valeurs de la boxe
            score = out_scores[i] #enregistre le score de prediction

            label = '{} {:.2f}'.format(classes_predict,score) # enregistre la classe et le score (pour le score on garde 2 chiffre aprés la virgule
            draw = ImageDraw.Draw(img) # va permettre de "dessiner" sur la vidéo
            size_label = draw.textsize(label, font) #définit la taille du label

            top, left, bottom, right = box # enregistre les valeurs de la boxe
            #transformation des valeurs
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(img.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom)) #permet d'afficher le nom de la classe , le score de sa prediction, les valeurs de position de la boxe
            leftvec.append(left)
            topvec.append(top)
            rightvec.append(right)
            bottomvec.append(bottom)

            #permet de calculer l'endroit où l'on va écrire les informations de la box
            if top - size_label[1] >=0:
                text_origin = np.array([left, top - size_label[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thick):
                draw.rectangle([left + i, top + i, right - i, bottom - i],outline=self.colors[p]) #création de la box sur l'image avec la couleur de la classe
            draw.rectangle([tuple(text_origin), tuple(text_origin + size_label)],fill=self.colors[p])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font) #permet d'écrire le text (label) à la position voulue (text_origin)
            del draw  #supression de draw


        end = timer() #prends la valeur du timer
        print(end-start)# affiche la différence de temps
        return img , leftvec, topvec, rightvec, bottomvec


    # permet de lire le fichier avec les classes et d'enregistrer les valeurs
    def _getclass(self):
        classes_path = os.path.expanduser(self.classes_path) #enregistre le nom des fichiers des classes
        with open(classes_path) as f: #ouverture du fichier et lecture gràce à la commande f
            name = f.readlines()     #lecture des lignes du fichier composé des classes
        name_vec = [i.strip() for i in name] #création d'un vecteur qui enregistre les noms des classes
        return name_vec

    def close(self):
        self.sess.close()  # permet de fermer la session
