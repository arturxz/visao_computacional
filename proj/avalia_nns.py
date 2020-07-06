"""
# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")
"""

# BIBLIOTECAS BASICAS
import csv
import cv2
import os
import numpy as np

# PARA MOSTRAR IMAGENS
import matplotlib.pyplot as mp

# PARA IA
import tensorflow as tf
from tensorflow import keras
from skimage import io, transform

# PARA DATA
from datetime import datetime

# CAMINHO PARA PASTAS
PATH = "dataset_train/"

# SHAPE PARA VGG16
shape_final = [224, 224, 3]

# SEPARACOES DE IMAGENS PARA TREINO E AVALIACAO
QTD_ITENS_TREINO = 3000
QTD_ITENS_AVALIA = 3634

def retorna_janela( img ):
    xi = -1
    xf = -1
    yi = -1
    yf = -1

    imax = img.shape[0]
    jmax = img.shape[1]

    # PIXEL MAIS ALTO DA JANELA
    for i in range( 0, imax ):
        if( 255 in img[i, :] ):
            xi = i
            break
    
    # PIXEL MAIS BAIXO DA JANELA
    for i in range( imax-1, -1, -1 ):
        if( 255 in img[i, :] ):
            xf = i
            break
    
    # PIXEL MAIS À ESQUERDA DA JANELA
    for j in range( 0, jmax ):
        if( 255 in img[:, j] ):
            yi = j
            break
    
    # PIXEL MAIS À DIREITA DA JANELA
    for j in range( jmax-1, -1, -1 ):
        if( 255 in img[:, j] ):
            yf = j
            break
    
    return xi, yi, xf, yf

def retorna_imagem_mascarada( img ):
    path_img = PATH + "image/" +img
    path_msk = PATH + "mask/" +img

    img  = io.imread( path_img )
    mask = io.imread( path_msk )

    # RETORNA xi, yi, xf, yf
    xi, yi, xf, yf = retorna_janela( mask )

    mask = mask / 255

    nimg = np.zeros( (xf-xi, yf-yi), np.uint8 )
    nmsk = np.zeros( (xf-xi, yf-yi), np.uint8 )

    nimg = img[xi:xf, yi:yf]
    nmsk = mask[xi:xf, yi:yf]

    res_nimg = nmsk * nimg

    mask = None
    nmsk = None
    img  = None
    nimg = None

    return res_nimg

def retorna_imagem_mascarada_redimensionada( path_img ):

    img_rgb = np.zeros( shape_final, np.uint8 )
    img = np.zeros( ( shape_final[0], shape_final[1] ), np.uint8 )

    img = cv2.resize( retorna_imagem_mascarada( path_img ), ( shape_final[1], shape_final[0] ) )
    
    #img = cv2.resize( path_img, (shape_final[1], shape_final[0]), interpolation=cv2.INTER_NEAREST )
    #img = transform.resize( path_img, shape_final )

    img_rgb[:,:,0] = img[:,:]
    img_rgb[:,:,1] = img[:,:]
    img_rgb[:,:,2] = img[:,:]

    return img_rgb

def avalia_model_inceptionResNetV2( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/inceptionResNetV2")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "inceptionResNetV2", lista_itens, lista_labels )

def avalia_model_inceptionv3( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/inceptionv3")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "inceptionv3", lista_itens, lista_labels )

def avalia_model_mobilenet( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/mobilenet")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "mobilenet", lista_itens, lista_labels )

def avalia_model_mobilenetv2( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/mobilenetv2")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "mobilenetv2", lista_itens, lista_labels )

def avalia_model_ResNet101V2( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/ResNet101V2")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "ResNet101V2", lista_itens, lista_labels )

def avalia_model_VGG16( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/vgg16")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "VGG16", lista_itens, lista_labels )


def avalia_model_VGG19( lista_itens, lista_labels ):
    # CARREGANDO MODELO ANTERIOR
    modelo_carregado = False
    try:
        modelo = keras.models.load_model("modelos_salvos/treino_dataset_completo/vgg19")
        modelo_carregado = True
    except Exception:
        modelo_carregado = False
        print( "Erro na carga do modelo. Carregando imagens de treio e treinando modelo." )
        print( "O modelo treinado será salvo." )

    # AVALIANDO MODELO
    if( modelo_carregado ):
        avalia_modelo( modelo, "VGG19", lista_itens, lista_labels )

def avalia_modelo( modelo, nome_modelo, lista_itens_avalia, lista_labels_avalia ):
    print( " ### Avaliando modelo:", nome_modelo )
    avalia_loss, avalia_acc = modelo.evaluate( lista_itens_avalia, lista_labels_avalia, verbose=2 )
    print( "   -- Acc: ", avalia_acc )
    print( "   -- Loss:", avalia_loss )

"""
    ######################################
    ##         MAIN COMEÇA AQUI         ##
    ######################################
"""

# ALOCANDO LISTAS DE TUMORES MALIGNOS E BENIGNOS
lista_itens = []
lista_itens_treino = []
lista_labels_treino = []

with open( "dataset_train/train.csv" ) as csv_file:
    # ABRINDO CSV
    csv_reader = csv.reader( csv_file, delimiter="," )

    # POPULANDO DICTIONARYS
    for row in csv_reader:
        if( row[0] != "ID" ):
            lista_itens.append( [ row[0], row[1] ] )
    
    print( " ## -- SEPARANDO IMAGENS -- ##" )
    for i in range( 0, len( lista_itens ) ):
        lista_itens_treino.append( lista_itens[ i ] )

lista_itens.clear()

print( " ## -- CRIANDO ARRAY DE ITENS PARA TREINO -- ##" )
for item in lista_itens_treino:
    lista_itens.append( retorna_imagem_mascarada_redimensionada( item[0] ) )
    lista_labels_treino.append( int( item[1] ) )

lista_itens_treino  = np.asarray( lista_itens ).astype( np.float32 ) / 255
lista_labels_treino = np.asarray( lista_labels_treino ).reshape( len(lista_labels_treino), 1 )
lista_itens.clear()

avalia_model_inceptionResNetV2( lista_itens_treino, lista_labels_treino )
avalia_model_inceptionv3( lista_itens_treino, lista_labels_treino )
avalia_model_mobilenet( lista_itens_treino, lista_labels_treino )
avalia_model_mobilenetv2( lista_itens_treino, lista_labels_treino )
avalia_model_ResNet101V2( lista_itens_treino, lista_labels_treino )
avalia_model_VGG16( lista_itens_treino, lista_labels_treino )
avalia_model_VGG19( lista_itens_treino, lista_labels_treino )