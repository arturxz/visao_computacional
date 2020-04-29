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
		return None

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

def dft( arr ):
	if( isinstance( arr, np.ndarray ) ):
		# GARANTE QUE O ARRAY É FLOAT
		x = np.asarray( arr, np.complex )
		
		# PREPARA OS VALORES DA SOMATORIA
		N = arr.shape[0]
		n = np.arange( N )
		k = n.reshape( ( N, 1) )
		M = np.exp( -2j * np.pi * k * n / N )
		
		# RETORNA A MULTIPLICACAO FINAL
		return np.dot( M, x )


def fft( arr ):
	if( isinstance( arr, np.ndarray ) ):
		x = np.asarray( arr, np.complex )
		N = x.shape[0]

		if( N % 2 != 0 ):
			raise ValueError( "Precisa ser potência de 2" )
		elif( N <= 2 ):
			return dft( arr ) # CASO BASE DA RECURSAO
		else:
			X_pares = fft( x[ ::2 ] ) # COMPUTA OS PARES
			X_impar = fft( x[ 1::2 ] ) # COMPUTA OS IMPARES
			terms = np.exp( -2j * np.pi * np.arange( N ) / N ) # PREPARA O EXPONENCIAL

			fu = X_pares + terms[ :int( N/2 ) ] * X_impar # CALCULA f(u)
			fu_k = X_pares + terms[ int( N/2 ): ] * X_impar # CALCULA f(u) + k

			return np.concatenate( [ fu, fu_k ] )

def fft_2d( arr ):
	print( "fft_2d" )
	if( isinstance( arr, np.ndarray ) ):
		arr_fourrier = np.asarray( arr, np.complex )
		if( nchannels( arr ) == 1 ):
			for i in range( arr.shape[ 0 ] ):
				arr_fourrier[i, :] = fft( arr[i, :] ) # FAZENDO PARA EIXO X
			
			for j in range( arr.shape[ 1 ] ):
				arr_fourrier[:, j] = fft( arr_fourrier[:, j] ) # FAZENDO PARA EIXO Y
		elif( nchannels( arr ) == 3 ):
			for canal in range( 3 ):
				arr_fourrier[:, :, canal] = fft_2d( arr[:, :, canal] ) # FAZENDO PARA CADA CANAL DE COR
		
		return arr_fourrier

def ifft( arr_fourrier ):
	if( isinstance( arr_fourrier, np.ndarray ) ):
		# GARANTINDO QUE O ARRAY EH DE COMPLEXOS
		arr_fourrier = np.asarray( arr_fourrier, np.complex )

		# RETORNA A CONJUGAÇÃO DA PARTE IMAGINARIA
		arr_fourrier_conjugate = np.conjugate( arr_fourrier )
		
		# CALCULA O FOURRIER DO ARRAY CONJUGADO
		arr = fft( arr_fourrier_conjugate )
		
		# RECONJUGANDO
		arr = np.conjugate( arr )

		# DIVISAO POR M
		arr = arr / arr_fourrier.shape[0]
		return arr

def ifft_2d( arr_fourrier ):
	print( "ifft_2d" )
	if( isinstance( arr_fourrier, np.ndarray ) ):
		arr = np.asarray( arr_fourrier, np.complex )
		if( nchannels( arr_fourrier ) == 1 ):
			for i in range( arr_fourrier.shape[0] ):
				arr[i, :] = ifft( arr_fourrier[i, :] ) # FAZENDO PARA EIXO X
			
			for j in range( arr_fourrier.shape[1] ):
				arr[:, j] = ifft( arr[:, j] ) # FAZENDO PARA EIXO Y
		elif( nchannels( arr_fourrier ) == 3 ):
			for canal in range( 3 ):
				arr[:, :, canal] = ifft_2d( arr[:, :, canal] ) # FAZENDO PARA CADA CANAL DE COR
		
		return arr.astype( np.float )
		

""" 
	Main
"""
listaImagens = retornaListaArquivos()
path = listaImagens[0]

img1 = imread( path )
img2 = fft_2d( img1 )

ifft_img1 = ifft_2d( img1 )
ifft_img2 = ifft_2d( img2 )

img1 = img1.astype( np.uint8 )
img2 = img2.astype( np.uint8 )

ifft_img1 = ifft_img1.astype( np.uint8 )
ifft_img2 = ifft_img2.astype( np.uint8 )

compare_images( img1, img2, "Original", "Fourrier" )
compare_images( ifft_img1, ifft_img2, "Fourrier Inversa de Original", "Fourrier Inversa de Fourrier" )

