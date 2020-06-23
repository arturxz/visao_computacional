import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from datetime import datetime

#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
#from keras.preprocessing.image import ImageDataGenerator

# CARREGANDO IMAGENS
trdata = keras.preprocessing.image.ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="cats_and_dogs_filtered/train",target_size=(224,224))
tsdata = keras.preprocessing.image.ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="cats_and_dogs_filtered/validation", target_size=(224,224))

# CAMADAS
model = keras.Sequential([
    keras.layers.Conv2D( input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=64,  kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2) ),
    keras.layers.Conv2D( filters=128, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=128, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2) ),
    keras.layers.Conv2D( filters=256, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=256, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=256, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2) ),
    keras.layers.Conv2D( filters=512, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=512, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=512, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2) ),
    keras.layers.Conv2D( filters=512, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=512, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.Conv2D( filters=512, kernel_size=(3,3), padding="same", activation="relu" ),
    keras.layers.MaxPool2D( pool_size=(2,2),strides=(2,2) ),
    keras.layers.Flatten(),
    keras.layers.Dense( units=4096,activation="relu" ),
    keras.layers.Dense( units=4096,activation="relu" ),
    keras.layers.Dense( units=2, activation="softmax" )
])

# COMPILANDO KERNEL
#from keras.optimizers import Adam
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#model.compile(
#    optimizer='adam',
#    loss='sparse_categorical_crossentropy',
#    metrics=['accuracy']
#)

# PRINTANDO MODELO
model.summary()

# CONFIGURANDO PRA SALVAR ESTADO ATUAL DA NN
#from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = keras.callbacks.ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

print( traindata )

# TREINANDO NN
model.fit( traindata, testdata, epochs=100 )
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    # train your model, then:
    savePath = saver.save(sess, 'vgg_16.ckpt')
#hist = model.fit( traindata, testdata, epochs=100, callbacks=[checkpoint,early] )
#hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])

# PREPARANDO AMOSTRAGEM
"""
import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
"""

# TESTANDO COM IMAGEM-TESTE
from keras.preprocessing import image
img = image.load_img("Pomeranian_01.jpeg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')