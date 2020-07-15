import os
import cv2
import sys
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm
from models import *
from pre_processing import preprocessa_dataset

from itertools import chain
from skimage.io import imread, imshow, imsave, imread_collection, concatenate_images
from skimage.filters import threshold_mean
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
TRAIN_PATH = 'dataset/train/'
TEST_PATH = 'dataset/evaluate/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

print( " ### SE HOUVER GPU, HABILITA ALOCAÇÃO DINAMICA DE MEMORIA" )
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def compare_images( img1, img2, img3=None, sub1="Imagem 1", sub2="Imagem 2", sub3="Imagem 3", titulo="Comparando Imagens" ):
    print( "-- compare" )
    ncols = 0
    
    img1_show = np.empty( img1.shape )
    img2_show = np.empty( img2.shape )
    img3_show = None

    if( int( np.max( img1 ) ) <= 1 ):
        print( "-- img1 [0,1]" )
        img1_show = np.asarray( img1*255, np.uint8 )
    else:
        img1_show = img1
    
    if( int( np.max( img2 ) ) <= 1 ):
        print( "-- img2 [0,1]" )
        img2_show = np.asarray( img2*255, np.uint8 )
    else:
        img2_show = img2

    if( img3 is None ):
        ncols = 2
    else:
        print( "-- img3 [0,1]" )
        ncols = 3
        img3_show = np.empty( img3.shape )
        if( int( np.max( img3 ) ) <= 1 ):
            img3_show = np.asarray( img3*255, np.uint8 )
        else:
            img3_show = img3

    plt.subplot( 1, ncols, 1 )
    plt.suptitle( titulo )
    plt1 = plt.imshow( np.squeeze( img1_show ), cmap=plt.gray(), vmin=0, vmax=255 )
    plt1.set_interpolation('nearest')
    plt.title( sub1 )
    
    plt.subplot( 1, ncols, 2 )
    plt2 = plt.imshow( np.squeeze( img2_show ), cmap=plt.gray(), vmin=0, vmax=255 )
    plt2.set_interpolation('nearest')
    plt.title( sub2 )
    
    if( not( img3_show is None ) ):
        plt.subplot( 1, ncols, 3 )
        plt3 = plt.imshow( np.squeeze( img3_show ), cmap=plt.gray(), vmin=0, vmax=255 )
        plt3.set_interpolation('nearest')
        plt.title( sub3 )
    
    plt.show()

caminhos = [os.path.join(TRAIN_PATH+"image/", nome) for nome in os.listdir(TRAIN_PATH+"image/")]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
imgs_train = [arq for arq in arquivos if arq.lower().endswith(".png")]

caminhos = [os.path.join(TRAIN_PATH+"mask/", nome) for nome in os.listdir(TRAIN_PATH+"mask/")]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
mask_train = [arq for arq in arquivos if arq.lower().endswith(".png")]
sys.stdout.flush()

# PREPROCESSANDO IMAGENS E MASCARAS DE TREINO
X_train = np.zeros((len(mask_train), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(mask_train), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print( "Lendo imagens e mascaras do dataset de treino" )
for i in tqdm( range( len(imgs_train) ) ):
    img = imread( imgs_train[i], format="png" )
    img = resize( img, (IMG_WIDTH, IMG_HEIGHT), mode='constant', preserve_range=True )
    
    #mskb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    msk = imread( mask_train[i], format="png" )
    msk = resize( msk, (IMG_WIDTH, IMG_HEIGHT), mode='constant', preserve_range=True )
    msk = (msk != 0) * 255
    #msk = cv2.normalize(msk, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    X_train[i,:,:,0] = np.squeeze( img )
    Y_train[i,:,:,0] = msk

print( "Processando imagens" )
#X_train = preprocessa_dataset( X_train, IMG_WIDTH, IMG_HEIGHT )
X_train = cv2.normalize(X_train, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
Y_train = cv2.normalize(Y_train, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print( "Completo." )

caminhos = [os.path.join(TEST_PATH+"image/", nome) for nome in os.listdir(TEST_PATH+"image/")]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
imgs_test = [arq for arq in arquivos if arq.lower().endswith(".png")]

# PREPROCESSANDO IMAGENS DE TESTE
X_test = np.zeros((len(imgs_test), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
sys.stdout.flush()

print( "Lendo imagens do dataset de teste" )
for i in tqdm( range( len(imgs_test) ) ):
    img = imread( imgs_test[i], format="png" )
    img = resize( img, (IMG_WIDTH, IMG_HEIGHT), mode='constant', preserve_range=True )
    sizes_test.append( [img.shape[0], img.shape[1]] )
    X_test[i,:,:,0] = np.squeeze( img )

print( "Processando imagens" )
#X_test = preprocessa_dataset( X_train, IMG_WIDTH, IMG_HEIGHT )
X_test = cv2.normalize(X_test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print( "Completo." )

# PRINTANDO
print( "X Train shape", X_train.shape )
print( "X Train Min, Max", np.min(X_train), np.max(X_train) )
print( "Y Train shape", Y_train.shape )
print( "Y Train Min, Max", np.min(Y_train), np.max(Y_train) )
print( "X Test shape", X_test.shape )
print( "X Test Min, Max", np.min(X_test), np.max(X_test) )

# MOSTRANDO IMAGEM E MASCARA ALEATORIA DE TREINO DEPOIS DE PREPROCESSADA
ix = random.randint( 0, len( imgs_train ) )
compare_images( X_train[ix], np.squeeze( Y_train[ix] ), None, imgs_train[ix].split("/").pop(), mask_train[ix].split("/").pop(), "Exemplo de imagem e máscara de treino" )

# CARREGANDO MODELO
model = unet_cnn( IMG_WIDTH, IMG_HEIGHT )
#model = BCDU_net_D1( IMG_WIDTH, IMG_HEIGHT )
#model = unet_vgg16( IMG_WIDTH, IMG_HEIGHT )

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=tf.metrics.MeanIoU(num_classes=2) )
model.summary()

# TREINA E SALVA MODELO
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('modelos_salvos/model_unet_256x256_elu_semLambda.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, shuffle=True, batch_size=4, epochs=50, callbacks=[earlystopper, checkpointer] )

# CARREGA MODELO
model = load_model('modelos_salvos/model_unet_256x256_elu_semLambda.h5')

# FAZENDO PREDICAO PARA VALIDACAO COM TREINO E COM VALIDACAO
preds_val   = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test  = model.predict(X_test, verbose=1)

# APLICANDO THRESHOLD
preds_val_t = (preds_val > 0.5) * 255
preds_val_t = np.asarray( preds_val_t, np.uint8 )

preds_test_t = (preds_test > 0.5) * 255
preds_test_t = np.asarray( preds_test_t, np.uint8 )

# MOSTRANDO 4 VALIDACOES ALEATORIAS
ix = random.randint(0, len(preds_val_t-1))
compare_images( X_train[ix], Y_train[ix], preds_val_t[ix], imgs_test[ix].split("/").pop(), imgs_test[ix].split("/").pop(), "Predito", "Imagem, Mascara e Predicao de mascara 1" )

ix = random.randint(0, len(preds_val_t-1))
compare_images( X_train[ix], Y_train[ix], preds_val_t[ix], imgs_test[ix].split("/").pop(), imgs_test[ix].split("/").pop(), "Predito", "Imagem, Mascara e Predicao de mascara 2" )

ix = random.randint(0, len(preds_val_t-1))
compare_images( X_train[ix], Y_train[ix], preds_val_t[ix], imgs_test[ix].split("/").pop(), imgs_test[ix].split("/").pop(), "Predito", "Imagem, Mascara e Predicao de mascara 3" )

ix = random.randint(0, len(preds_val_t-1))
compare_images( X_train[ix], Y_train[ix], preds_val_t[ix], imgs_test[ix].split("/").pop(), imgs_test[ix].split("/").pop(), "Predito", "Imagem, Mascara e Predicao de mascara 4" )

# SALVANDO IMAGENS
print( "qtd imgs avaliacao:", len( imgs_test ) )
for i in range( len( preds_val_t ) ):
    imsave( TEST_PATH + "result/" + imgs_test[i].split("/").pop(), preds_test_t[i] )
