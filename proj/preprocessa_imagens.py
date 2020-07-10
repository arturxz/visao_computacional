from matplotlib.pyplot import imread
from tqdm import tqdm
import cv2
import os

dir_treino_imgs_entrada = "datasets/dataset_train/image/"
dir_treino_mask_entrada = "datasets/dataset_train/mask/"

dir_treino_imgs_saida = "datasets/dataset_train/image_pre/"
dir_treino_mask_saida = "datasets/dataset_train/mask_pre/"

caminhos = [os.path.join(dir_treino_imgs_entrada, nome) for nome in os.listdir(dir_treino_imgs_entrada)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
imgs = [arq for arq in arquivos if arq.lower().endswith(".png")]

caminhos = [os.path.join(dir_treino_mask_entrada, nome) for nome in os.listdir(dir_treino_mask_entrada)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
mask = [arq for arq in arquivos if arq.lower().endswith(".png")]

"""
filename = imgs[0].split("/")
filename = filename.pop()
img = imread( dir_treino_imgs_entrada + filename, format="png" )
print( "Abre:", img.shape )
img = cv2.resize( img, (224, 224), interpolation=cv2.INTER_CUBIC )
print( "Depois do esize:", img.shape )
cv2.imwrite( dir_treino_imgs_saida + filename.lower(), img )
"""

print( "Pré-processando Imagens" )
for i in tqdm( range( len(imgs) ) ):
    filename = imgs[i].split("/")
    filename = filename.pop()
    img = imread( dir_treino_imgs_entrada + filename, format="png" )
    img = cv2.resize( img, (128, 128), interpolation=cv2.INTER_CUBIC )
    #imsave( dir_treino_imgs_saida + filename.lower(), img, format="png" )
    cv2.imwrite( dir_treino_imgs_saida + filename.lower(), img )

print( "Pré-processando Máscaras" )
for i in tqdm( range( len(imgs) ) ):
    filename = imgs[i].split("/")
    filename = filename.pop()
    img = imread( dir_treino_mask_entrada + filename, format="png" )
    img = cv2.resize( img, (128, 128), interpolation=cv2.INTER_CUBIC )
    #imsave( dir_treino_mask_saida + filename.lower(), img, format="png" )
    cv2.imwrite( dir_treino_mask_saida + filename.lower(), img )
