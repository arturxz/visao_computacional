import os
import platform
import numpy as np
import time as t
import matplotlib.pyplot as mp

from PIL import Image
from math import floor

def retornaListaArquivos( path=None ):
	
	isLinux = "linux" == platform.system().lower()
	
	if( path == None ):
		if( isLinux ):
			path = os.getcwd() + "/img"
		else:
			path = os.getcwd() +"\\img"
	
	lista = os.listdir( path )
	
	for i in range(0, len(lista)):
		if( isLinux ):
			lista[i] = path +"/" +lista[i]
		else:
			lista[i] = path +"\\" +lista[i]
	return lista

def imread( filename ):
	img = mp.imread( filename, np.uint8 )
	return img

def is_gray_image( img ):
	if( isinstance(img, np.ndarray) ):
		if( img.ndim == 2 ):
			return True
		else:
			return False
	else:
		return False

def nchannels( img ):
	if( isinstance( img, np.ndarray ) ):
		if( img.ndim == 2 ):
			return 1
		else:
			return img.shape[2]
	else:
		return -1

def imshow( img ):
	plt = mp.imshow(img, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt.set_interpolation('nearest')
	mp.show()

def compare_images( img1, img2, sub1="Imagem 1", sub2="Imagem 2"):
	mp.subplot( 2, 1, 1 )
	plt1 = mp.imshow(img1, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt1.set_interpolation('nearest')
	mp.title( "Comparando Imagens" )
	mp.ylabel( sub1 )
	
	mp.subplot( 2, 1, 2 )
	plt2 = mp.imshow(img2, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt2.set_interpolation('nearest')
	mp.ylabel( sub2 )

	mp.show()

def rgb2gray( img ):
	print("method: rgb2gray")
	if( isinstance( img, np.ndarray ) ):
		if( is_gray_image( img ) ):
			return img
		else:
			r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
			gray = 0.299 * r + 0.587 * g + 0.114 * b
			return np.asarray(gray, dtype=np.uint8)
	return None

