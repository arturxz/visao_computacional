import os
import platform
import numpy as np
import time as t
import matplotlib.pyplot as mp
import PIL

from skimage import io
from math import floor, ceil, log

def retornaListaArquivos( path=None ):
	
	isLinux = "linux" == platform.system().lower()
	
	if( path == None ):
		if( isLinux ):
			path = os.getcwd() + "/dataset_train/01"
		else:
			path = os.getcwd() +"\\dataset_train\\01"
	
	lista = os.listdir( path )
	
	for i in range(0, len(lista)):
		if( isLinux ):
			lista[i] = path +"/" +lista[i]
		else:
			lista[i] = path +"\\" +lista[i]
	return lista

def imread( filename ):
	#img = PIL.Image.open( filename )
	#return img
    return io.imread( filename )

def imsave( img, filename ):
    img.save( filename )

def imshow( img ):
	plt = mp.imshow(img, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt.set_interpolation('nearest')
	mp.show()

def compare_images( img1, img2, sub1="Imagem 1", sub2="Imagem 2", titulo="Comparando Imagens" ):
	mp.subplot( 1, 2, 1 )
	plt1 = mp.imshow(img1, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt1.set_interpolation('nearest')
	mp.title( titulo )
	mp.ylabel( sub1 )
	
	mp.subplot( 1, 2, 2 )
	plt2 = mp.imshow(img2, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt2.set_interpolation('nearest')
	mp.ylabel( sub2 )

	mp.show()

def is_gray_image( img ):
	if( isinstance(img, np.ndarray) ):
		if( img.ndim == 2 ):
			return True
		else:
			return False
	else:
		return False

def thresh( img, tr ):
	print("method: threshold")
	print("thresh:", tr)
	if( isinstance( img, np.ndarray ) ):
		if( is_gray_image( img ) ):
			img_thresh = (img >= tr) * 255
			return img_thresh
		else:
			img_bool_r = img[:, :, 0] >= tr
			img_bool_g = img[:, :, 1] >= tr
			img_bool_b = img[:, :, 2] >= tr
			img_thresh = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), np.uint8 )
			
			img_thresh[:, :, 0] = img_bool_r * 255
			img_thresh[:, :, 1] = img_bool_g * 255
			img_thresh[:, :, 2] = img_bool_b * 255
			
			return img_thresh
	return None

listaImagens = retornaListaArquivos( None )
path = listaImagens[0]

img = imread( path )
print( img.shape )

compare_images( img, thresh(img, 20), "Original", "Thresh" )
