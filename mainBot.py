import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import random
import pickle

stemmer = LancasterStemmer()

#Cargamos el archivo json con los patrones y respuestas:
with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

try:
    with open("variables.pickle", "rb") as archivoPicke:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPicke)
except:
    palabras = []
    tags = []
    aux1 = []
    aux2 = []

    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            #Separa oraciones en palabras:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            aux1.append(auxPalabra)
            aux2.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w!= "?" or w != "¿" or w != "¡" or w !="!"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []

    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(aux1):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento if w!= "?" or w != "¿" or w != "¡" or w !="!"]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(aux2[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = np.array(entrenamiento)
    salida = np.array(salida)
    with open("variables.pickle", "wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

tf.compat.v1.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 6)
red = tflearn.fully_connected(red, 6)

red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")

red = tflearn.regression(red)

modelo = tflearn.DNN(red)

try:
    modelo.load("modelo.tflearn")
except:
    modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=6, show_metric=True)
    modelo.save("modelo.tflearn")

def mainBot():
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = modelo.predict([np.array(cubeta)])
        resultadosIndices = np.argmax(resultados)
        tag = tags[resultadosIndices]

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]
        
        print("BOT:", random.choice(respuesta))
mainBot()