import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

filepath = os.path.dirname(os.path.abspath(__file__))

# Sección 1: Cargando las imágenes de los directorios para entrenar y probarlos.
# Lineas 15 a 53: Tenemos nuestras imágenes de entrenamiento y pruebas almacenados
# en directorios. Para cargar esas imágenes con el proposito de entrenamiento
# y validación, usamos una clase muy poderosa llamada ImageDataGenerator, dada
# por Keras.
training_img_dir = filepath + '/chest_xray/train'
test_img_dir = filepath + '/chest_xray/test'

# La clase ImageDataGenerator proporciona un mecánismo para cargar ya sea 
# conjuntos de datos pequeños o grandes.
# Se le da la instrucción a ImageDataGenerator de escalar para normalizar los 
# valores de un rango range(0,1).
#
# La siguiente instrucción inicializa la clase ImageDataGenerator. Le pasamos
# el argumento rescale=1./255 porque queremos normalizar los valores de pixeles
# para que estén entre 0 y 1. Esta normalización está hecha multiplicando cada pixel
# de las imágenes por 1/255. Llamamos a esta linea datagen.
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Crea un entrenador de iteración de imágenes que será cargado en un pequeño batch
# Hace un 'resize' a todas las imágenes a un tamaño estándar.
#
# Esta línea llama a la función flow_from_directory() del objeto datagen.
# Esta función carga carga imágenes del directorio de training_img_dir a modo
# de lote (batch_size=8), y redimensiona las imágenes al tamaño indicado en target_size(1024,1024).
# Esta es una función altamente escalable y puede ser capaz de cargar millones de imágenes
# sin cargarlas todas en memoria. Las va a cargar dependiendo lo que se mande en batch_size.
# Redimensionar todas las imágenes a un tamaño estándar es importante para muchos algoritmos ml.
# El valor por defecto de redimensión es de 256x256
train_it = datagen.flow_from_directory(training_img_dir, batch_size=8, target_size=(1024, 1024))

# Crea un entrenador de iteración de imágenes que será cargado en un pequeño batch
# Hace un 'resize' a todas las imágenes a un tamaño estándar.
# La siguiente instrucción hace lo mismo que la anterior con la excepción de que carga las
# imágenes del directorio de prueba.
# A pesar de que tenemos datos de validación en nuestro directorio, el número es pequeño, y 
# por lo tanto hemos decidido usar el conjunto de datos de prueba para validación.
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
    
    # Crea una Red Neuronal Secuencial la cual le apilamos capas.
    model = tf.keras.models.Sequential()
    # La función add() es usada para añadir capas en un orden secuencial.
    # Añadimos nuestra primera capa a la red. La primera capa debe ser una 
    # capa convolutiva que toma la entrada (valores de píxeles de la imagen).
    # Aquí estamos usando la clase Conv2D para definir nuestra capa convolutiva.
    # Estamos pasando 5 parámetros importantes a Conv2D():
    #   - filters: que en este ejemplo es 32
    #   - kernel dimension: que en este ejemplo es 3x3 píxeles
    #   - función de activación: en este caso es relu (ya que los valores de
    #     los píxeles van de 0 a 1 y nunca son negativos)
    #   - strides: Por defecto es (1,1), pero le pusimos (2,2)
    #   - input_shape: Ya que nuestras imágenes están redimensionadas a 1024x1024
    #     píxeles a color (con 3 canales), entonces, input_shape es (1024,1024,3)
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', strides=(2,2),
                input_shape=(1024, 1024, 3)))
    # Añade la capa de pooling (agrupación). Recuerda que las capas
    # convolutivas y de agrupación están alternadas y vienen en pares, excepto
    # por la capa antes de las capas MLP (MultiLayer Perceptron). Estamos pasando el 
    # argumento para establecer el tamaño de la rejilla del kernel. En este ejemplo es
    # (2,2)
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    # Las siguientes líneas son de nuevo capas convolutivas y de agrupamiento.
    # Podemos colocar tantas como sean requeridas para conseguir los niveles de 
    # exactitud deseados.
    model.add(tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    # La salida de la siguiente capa de convolución, es alimentada a la primera
    # capa de MLP. Recuerda que la primera capa de MLP es llamada la capa
    # de entrada, seguida de capas ocultas, y finalmente, la capa de salida.
    model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), activation='relu'))
    # La siguiente instrucción aplana la salida de la instrucción anterior
    model.add(tf.keras.layers.Flatten())
    # La siguiente instrucción es la capa oculta del MLP (Perceptron multicapa)
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # La siguiente es la última capa. Como se explicó anteriormente, 
    # estamos usando la función de activación softmax ya que estamos resolviendo
    # un problema de clasificación que involucra dos clases.
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
model_path = filepath + '/models/pneumiacnn'
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