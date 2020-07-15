import numpy as np
import cv2

#from help_functions import *


# SEQUENCIA DE PREPROCESSAMENTO DO DATASET
def preprocessa_dataset( arr_imgs, final_width=256, final_height=256 ):
    #imgs = dataset_normalized( arr_imgs )
    #print( "  -- Max1:", np.max( arr_imgs ) )
    #print( "  -- Min1:", np.min( arr_imgs ) )
    #imgs = cv2.normalize(arr_imgs, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print( "  -- Max1:", np.max( arr_imgs ) )
    print( "  -- Min1:", np.min( arr_imgs ) )
    imgs = clahe_equalized( arr_imgs )
    print( "  -- Max2:", np.max( imgs ) )
    print( "  -- Min2:", np.min( imgs ) )
    imgs = imgs = cv2.normalize(arr_imgs, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #imgs = resize( imgs, final_width, final_height )
    #imgs = imgs / 255
    print( "  -- Max3:", np.max( imgs ) )
    print( "  -- Min3:", np.min( imgs ) )
    return imgs

"""
    Equalizacao Adaptativa de Histograma Limitada ao Contraste (CLAHE)
    1 - A imagem é dividida em pequenos blocos (padrao 8x8)
    2 - Se o histograma estiver acima do limite de contraste (padrão 40), os pixels são distribuidos para os blocos vizinhos.
    3 - Em cada um dos blocos é feita equalização de histograma
    4 - É feita interpolação bilinear nas bordas dos blocos para eliminar artefatos.
"""
def clahe_equalized( imgs ):
    print("-- clahe")
    
    imgs_equalized = np.empty( imgs.shape )
    for i in range(imgs.shape[0]):
        # APLICANDO EQUALIZACAO EM CADA UMA DAS IMAGENS
        clahe = cv2.createCLAHE( clipLimit=2.0, tileGridSize=(8,8) )
        imgs_equalized[i,:,:,0] = clahe.apply( imgs[i,:,:,0].astype( np.uint8 ) )

    return imgs_equalized

# NORMALIZANDO EM RELACAO AO DATASET INTEIRO
def dataset_normalized( imgs ):
    print("-- norm")
    imgs_norm       = np.empty( imgs.shape )
    desvio_padrao   = np.std( imgs )
    media_imgs      = np.mean( imgs )

    # APLICANDO GLOBALMENTE
    imgs_norm = ( imgs - media_imgs ) / desvio_padrao

    # APLICANDO INDIVIDUALMENTE A CADA IMAGEM
    for i in range( imgs.shape[0] ):
        imgs_norm[i] = ( ( imgs_norm[i] - np.min(imgs_norm[i] ) ) / ( np.max( imgs_norm[i] ) - np.min( imgs_norm[i] ) ) ) * 255

    return imgs_norm
