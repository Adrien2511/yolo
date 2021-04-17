import cv2
import time
import numpy as np
import os
from keras.models import Model, load_model
from keras_preprocessing import image
from keras_applications.mobilenet import preprocess_input
from PIL import Image
from yolo import YOLO
font = cv2.FONT_HERSHEY_DUPLEX
import tensorflow as tf

def localisation_objet():

    chemin_model = "model_data/yolo.h5"  #on encode les chemin d'accès au modèle et à la vidéo "test"
    chemin_video = "videotest.avi"

    print(f"Running Video {chemin_video} for model {chemin_model}") #affichage du contenu à l'écran

    classes = "model_dat/coco_classes.txt" #on se base sur le modèle "coco" pour nos différentes classes

    # enregistrement des cadres ("anchors") --> contour des box
    # choix du modèle
    if "tiny" in chemin_model: # tiny = modèle moins complet --> plus rapide --> moins performant
        anchors_yolo="model_data/tiny_yolo_anchors.txt"
    else:
        anchors_yolo="model_data/yolo_anchors.txt"

    #enregistrement du modèle
    yolo_model = YOLO(classes_path1=classes, anchors_path1=anchors_yolo, model_path1=chemin_model) #on passe les paramètres pour le fichier "yolo.py"

    capture = cv2.VideoCapture(chemin_video) #ouverture webcam
    #capture=cv2.VideoCapture(chemin_video) #ouverture video "test"
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   #initialisation de la largeur de la video : 640px
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  #initialisation de la hauteur de la video : 360px

    #Déclaration de plusieurs variables

    intervalle_affichage_fps = 1 #1 seconde
    frequence_image = 0
    frequences_image = [] #création d'un vector
    compteur_image = 0
    temps_depart = time.time() #renvoie le nombre de secondes au départ
    temps_ecoule = 0

    while True:
        ouvert, image = capture.read()

        if ouvert==True:  #si la caméra est bien détectée --> True
            image_taille = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA) #INTER_AREA = rééchantillonnage en utilisant la relation de zone de pixel
            image_yolo = Image.fromarray(image_taille) #enregistrement de la taille de l'image comme une image de couleur grise
            r_image, left, top, right, bottom = yolo_model.image_detection(image_yolo) #permet de détecter l'image par rapport au modèle yolo utilisé
            print("left :",left,"top :",top,"right :", right, "bottom :",bottom)


            fin_temps = time.time() #renvoie le nombre de secondes à la fin

            if (fin_temps - temps_depart) > intervalle_affichage_fps:
                frequence_image = int(compteur_image / (fin_temps - temps_depart)) #[s]^-1 = freqence
                temps_ecoule += fin_temps - temps_depart
                compteur_image = 0
                frequences_image.append(frequence_image) #on remplit la liste à chaque passage dans la boucle if
                temps_depart = time.time()#renvoie le nombre de secondes au départ

            compteur_image += 1

            resultat = np.asarray(r_image) #conversion de la liste r_image en un tableau

            #permet de dessiner une chaîne de texte sur n'importe quelle image. Coordonnées sous forme de "tuples (x,y)" dans le coin inférieur gauche de l'image
            cv2.putText(resultat, str(frequence_image) + "fps", (500, 50),
                        font, 1, (0,  255, 0), thickness=2, lineType=2)

            #couleur de la chaine de texte : ici (255, 0, 0) = bleu
            cv2.putText(resultat, 'detection_objet', (10, 50), font,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('detection_objet', resultat) #affichage caméra

            if cv2.waitKey(1) & 0xFF == ord('q'): #si la caméra est ouverte et qu'on appuie sur la touche " q " --> fermeture du programme
                break
        else:
            break

    capture.release()  #quitter video
    cv2.destroyAllWindows() #fonction liée au clavier pour couper video à la fin

localisation_objet() #appel de la fonction
