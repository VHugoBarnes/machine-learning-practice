import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

filepath = os.path.dirname(os.path.abspath(__file__))

# Sección 1: Cargando las imágenes de los directorios para entrenar y probarlos
training_img_dir = filepath + '/chest_xray/train'
test_img_dir = filepath + '/chest_xray/test'

# La clase ImageDataGenerator proporciona un mecánismo para cargar ya sea 
# conjuntos de datos pequeños o grandes.
# Se le da la instrucción a ImageDataGenerator de escalar para normalizar los 
# valores de un rango range(0,1).
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Crea un entrenador de iteración de imágenes que será cargado en un pequeño batch
# Hace un 'resize' a todas las imágenes a un tamaño estándar.
train_it = datagen.flow_from_directory(training_img_dir, batch_size=8, target_size=(1024, 1024))

# Crea un entrenador de iteración de imágenes que será cargado en un pequeño batch
# Hace un 'resize' a todas las imágenes a un tamaño estándar.
test_it = datagen.flow_from_directory(test_img_dir, batch_size=8, target_size=(1024, 1024))

# Las siguientes 3 líneas son opcionales
# La función next() retorna los valores de los pixeles y etiquetas como arrays de NumPy
train_images, train_labels = train_it.next()
test_images, test_labels = test_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (train_images.shape,train_images.min(), train_images.max()))

# Sección 2
# Construir una red CNN y entrenarla con el conjunto de datos de entrenamiento
# Puedes pasar argumentos a build_cnn() para establecer algunos de los valores
# como nombre de filtros, strides, funciones activadoras, número de layers, etc.
def build_cnn():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', strides=(2,2),
                input_shape=(1024, 1024, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model

# Construye el modelo CNN
model = build_cnn()
# Compila el modelo con un optimizador y una _loss function_
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Ajusta el modelo. La función fit_generator() carga iterativamente grandes
# números de imagenes en lotes
history = model.fit_generator(train_it, 
                              epochs=10, 
                              steps_per_epoch=16, 
                              validation_data=test_it,
                              validation_steps=8)

# Sección 3
# Guarda la CNN en el disco para poder usarlo posteriormente
model_path = filepath + 'models/pneumiacnn'
model.save(filepath=model_path)

# Sección 4
# Muestra las metricas de evaluación
print(history.history.keys())
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)