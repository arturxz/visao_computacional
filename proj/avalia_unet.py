import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imsave, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

"""
    INICIALIZACOES E VARIAVEIS GERAIS
"""
# SETANDO PARAMETROS GERAIS
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
TEST_PATH = 'dataset/evaluate/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# FUNCAO QUE ABRE IMAGENS PARA COMPARACAO
def compare_images( img1, img2, img3=None, sub1="Imagem 1", sub2="Imagem 2", sub3="Imagem 3", titulo="Comparando Imagens" ):
    ncols = 0
    if( img3 is None ):
        ncols = 2
    else:
        ncols = 3
    
    plt.subplot( 1, ncols, 1 )
    plt.suptitle( titulo )
    plt1 = plt.imshow( np.squeeze( img1 ), cmap=plt.gray() )
    plt1.set_interpolation('nearest')
    plt.title( sub1 )
    
    plt.subplot( 1, ncols, 2 )
    plt2 = plt.imshow( np.squeeze( img2 ), cmap=plt.gray() )
    plt2.set_interpolation('nearest')
    plt.title( sub2 )
    
    if( not( img3 is None ) ):
        plt.subplot( 1, ncols, 3 )
        plt3 = plt.imshow( np.squeeze( img3 ), cmap=plt.gray() )
        plt3.set_interpolation('nearest')
        plt.title( sub3 )
    plt.show()

def habilita_alocacao_dinamica_gpu():
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

habilita_alocacao_dinamica_gpu()

caminhos = [os.path.join(TEST_PATH+"image/", nome) for nome in os.listdir(TEST_PATH+"image/")]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
imgs_test = [arq for arq in arquivos if arq.lower().endswith(".png")]

# Get and resize test images
X_test = np.zeros((len(imgs_test), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
sys.stdout.flush()

print( "Pré-processando dataset de teste" )
for i in tqdm( range( len(imgs_test) ) ):
    img = imread( imgs_test[i], format="png" )
    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize( img, (IMG_WIDTH, IMG_HEIGHT), mode='constant', preserve_range=True )
    X_test[i,:,:,0] = np.squeeze( img )

# Predict on train, val and test
model = load_model('model-unet-1.h5', custom_objects={'MeanIoU': tf.metrics.MeanIoU(num_classes=2)})

preds_test  = model.predict(X_test, verbose=1)

# Threshold predictions
preds_test_t = (preds_test > 0.5) * 255
preds_test_t = np.asarray( preds_test_t, np.uint8 )

print( "Salvando imagens de evaluate" )
for i in range( 0, len( imgs_test ) ):
    img = preds_test_t[i]
    img = resize( img, sizes_test[i], mode='constant', preserve_range=True )
    imsave( TEST_PATH + "result/" + imgs_test[i].split("/").pop(), img )

