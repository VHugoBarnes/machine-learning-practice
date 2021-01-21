import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

filepath = os.path.dirname(os.path.abspath(__file__))


model_path = filepath + '/models/pneumiacnn'
val_img_dir = filepath + '/chest_xray/val'

# ImageDataGenerator proporciona un mecánismo para cargar conjuntos de datos tanto
# pequeños o grandes
# Le damos la instrucción a ImageDataGenerator de escalar para normalizar 
# los valores de los píxeles para el range(0,1)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
# Crea un iterador de imágenes de entrenamiento que será cargado en un lote 
# pequeño. Redimensiona todas las imágenes a un tamaño estándar.
val_it = datagen.flow_from_directory(val_img_dir, batch_size=8, target_size=(1024,1024))

# Carga y crea el modelo exacto, incluyendo los pesos y el optimizador
model = tf.keras.models.load_model(model_path)

# Predice la clase de la imagen de entrada al modelo cargado
predicted = model.predict_generator(val_it, steps=24)
print('Predicted', predicted)