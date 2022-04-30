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


palabras = []
tags = []
aux1 = []
aux2 = []
#Separamos en palabras:
for contenido in datos["contenido"]:
	for patrones in contenido["patrones"]:
		#Separa oraciones en palabras:
		auxPalabra = nltk.word_tokenize(patrones)
		palabras.extend(auxPalabra)
		aux1.append(auxPalabra)
		aux2.append(contenido["tag"])

		if contenido["tag"] not in tags:
			tags.append(contenido["tag"])
#Simplificamos el texto para que sea mas facil analizarlo:
palabras = [stemmer.stem(w.lower()) for w in palabras if w!= "?" or w != "¿" or w != "¡" or w !="!"]
palabras = sorted(list(set(palabras)))
tags = sorted(tags)

entrenamiento = []
salida = []

salidaVacia = [0 for _ in range(len(tags))]

#Usamos el algoritmo de la cubeta:
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

#Cargamos el archivo pickle con toda la informacion:
entrenamiento = np.array(entrenamiento)
salida = np.array(salida)

#Reseteamos todas las neuronas por si acaso:
tf.compat.v1.reset_default_graph()

#Indicamos que queremos entrenar (capa inicial con la cantidad de palabras):
red = tflearn.input_data(shape=[None, len(entrenamiento[0])])

#Creamos una capa de neuronas (aqui depende de nuestra cantidad de palabras en TAGS del JSON):
red = tflearn.fully_connected(red, 123)

#Creamos otra capa de neuronas:
red = tflearn.fully_connected(red, 123)

#Esta sera la capa final, o sea, la cantidad de patrones que tengamos:
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")

#Regresamos los resultados:
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

#try: #Cargamos el modelo si ya tenemos las neuronas entrenadas
#modelo.load("modelo.tflearn")
#except: #En caso de que no las tengamos entrenadas creamos uno por primera vez:
modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=123, show_metric=True)
#modelo.save("modelo.tflearn")

#Funcion principal que reconoce lo que introduce el usuario y lo compara para dar una respuesta:
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