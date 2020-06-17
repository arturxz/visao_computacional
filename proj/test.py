# BIBLIOTECAS BASICAS
import csv
import cv2
import numpy as np

# PARA MOSTRAR IMAGENS
import matplotlib.pyplot as mp

# PARA IA
import tensorflow as tf
from tensorflow import keras
from skimage import io, transform

# IMAGENS PARA TREINO
QTD_ITENS_TREINO = 3000
QTD_ITENS_AVALIA = 3634

# SHAPE GERAL
#shape_maior = [-1, -1]
#shape_menor = [99999, 99999]
shape_final = [0, 0]

# CAMINHO PARA PASTAS
PATH = "dataset_train/"

def imshow( img ):
	plt = mp.imshow(img, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt.set_interpolation('nearest')
	mp.show()


def calcula_media_shapes( lst_imgs ):
    qtd_imgs = len( lst_imgs )
    wide = 0
    tall = 0

    for i in range(0, qtd_imgs ):
        wide = wide + lst_imgs[i].shape[0]
        tall = tall + lst_imgs[i].shape[1]
    
    wide = int( wide / qtd_imgs )
    tall = int( tall / qtd_imgs )

    shape_final[0] = wide
    shape_final[1] = tall

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

    # APLICANDO ABERTURA
    #res_nimg = cv2.morphologyEx(res_nimg, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    mask = None
    nmsk = None
    img  = None
    nimg = None

    return res_nimg

def retorna_listas_imagens_labels( lst_imgs ):
    lista_imgs = []
    lista_lbls = [] 

    print( "Tamanho:", len( lst_imgs ) )

    for item in lst_imgs:
        lista_imgs.append( retorna_imagem_mascarada( item[ 0 ] ) )
        lista_lbls.append( int( item[ 1 ] ) )
    
    return np.asarray( lista_imgs ), np.asarray( lista_lbls )

def retorna_imagens_redimensionadas( lista_imgs ):
    qtd_imagens = len( lista_imgs )
    
    arr_imgs = np.zeros( ( qtd_imagens, shape_final[0], shape_final[1] ), np.float32 )

    for i in range( 0, qtd_imagens ):
        #arr_imgs[i,:,:] = transform.resize( lista_imgs[i], shape_final )
        arr_imgs[i,:,:] = cv2.resize( lista_imgs[i], (shape_final[1], shape_final[0]) )
        #arr_imgs[i,:,:] = cv2.resize( lista_imgs[i], (shape_final[1], shape_final[0]), interpolation=cv2.INTER_NEAREST )
        lista_imgs[i] = None
    
    lista_imgs = None    
    return arr_imgs

"""
    ----------------------------------
    --       MAIN COMECA AQUI       --
    ----------------------------------
"""

# ALOCANDO LISTAS DE TUMORES MALIGNOS E BENIGNOS
lista_imagens = []
lista_imagens_treino = []
lista_imagens_avalia = []
lista_imagens_prediz = []

with open( "dataset_train/train.csv" ) as csv_file:
    # ABRINDO CSV
    csv_reader = csv.reader( csv_file, delimiter="," )

    # POPULANDO DICTIONARYS
    for row in csv_reader:
        if( row[0] != "ID" ):
            lista_imagens.append( [ row[0], row[1] ] )
    
    for i in range( 0, len( lista_imagens ) ):
        if( i < QTD_ITENS_TREINO ):
            lista_imagens_treino.append( lista_imagens[ i ] )
        elif( i < QTD_ITENS_AVALIA ):
            lista_imagens_avalia.append( lista_imagens[ i ] )
        else:
            lista_imagens_prediz.append( lista_imagens[ i ] )

# CARREGANDO DADOS PARA TREINO
print( "-- Carregando imagens de treino" )
lista_tumores_treino, lista_labels_treino = retorna_listas_imagens_labels( lista_imagens_treino )

# CALCULA MEDIA DOS SHAPES DAS IMAGENS
calcula_media_shapes( lista_tumores_treino )

# REDIMENSIONANDO IMAGENS
print( "-- Redimensionando Shapes Treino:" )
lista_tumores_treino = retorna_imagens_redimensionadas( lista_tumores_treino )
lista_tumores_treino = lista_tumores_treino / 255.0

print( "-- Shapes:", lista_tumores_treino.shape, lista_labels_treino.shape )
# CRIANDO ENCADEAMENTO DE CAMADAS DO MODELO
print( "-- Criando encadeamento das camadas do modelo" )
modelo = keras.Sequential([
    keras.layers.Flatten( input_shape=shape_final ),
    keras.layers.Dense( 256, activation='relu' ),
    keras.layers.Dense( 128, activation='relu' ),
    keras.layers.Dense( 64, kernel_initializer='lecun_normal', activation='selu'),
    keras.layers.Dense( 32, kernel_initializer='lecun_normal', activation='selu'),
    keras.layers.Dense( 2, activation='softmax' )
])

# COMPILANDO O MODELO
print( "-- Compilando Modelo" )
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print( "-- Efetuando treino do modelo" )
modelo.fit( lista_tumores_treino, lista_labels_treino, epochs=2 )

# LIMPANDO DADOS DE TREINO DA MEMORIA
lista_tumores_treino = None
lista_labels_treino = None

# EFETUANDO CARGA DE IMAGENS DE AVALIACAO
print( "-- Carregando imagens de Avaliação" )
lista_tumores_avalia, lista_labels_avalia = retorna_listas_imagens_labels( lista_imagens_avalia )

# REDIMENSIONANDO IMAGENS
print( "-- Redimensionando Shapes Avalia:" )
lista_tumores_avalia = retorna_imagens_redimensionadas( lista_tumores_avalia )
lista_tumores_avalia = lista_tumores_avalia / 255.0

print( "-- Efetuando Avaliação do modelo" )
avalia_loss, avalia_acc = modelo.evaluate( lista_tumores_avalia, lista_labels_avalia, verbose=2 )
print( "-- Teste de Acurácia:", avalia_acc )

# LIMPANDO DADOS DE AVALIACAO DA MEMORIA
lista_tumores_avalia = None
lista_labels_avalia = None

print( "-- Carregando imagens de Predição" )
lista_tumores_prediz, lista_labels_prediz = retorna_listas_imagens_labels( lista_imagens_prediz )

# REDIMENSIONANDO IMAGENS
print( "-- Redimensionando Shapes Avalia:" )
lista_tumores_prediz = retorna_imagens_redimensionadas( lista_tumores_prediz )
lista_tumores_prediz = lista_tumores_prediz / 255.0

print( "-- Efetuando predições com o modelo" )
predito = modelo.predict( lista_tumores_prediz )
for i in range( 0, len( predito ) ):
    print( "------------------------------" )
    print( "Item", i )
    print( "Predição: ", np.argmax( predito[ i ] ) )
    print( "Realidade:", lista_labels_prediz[ i ] )

