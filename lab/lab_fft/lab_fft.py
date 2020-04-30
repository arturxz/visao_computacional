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

def compare_images( img1, img2, sub1="Imagem 1", sub2="Imagem 2", titulo="Comparando Imagens"):
	mp.subplot( 2, 1, 1 )
	plt1 = mp.imshow(img1, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt1.set_interpolation('nearest')
	mp.title( titulo )
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
		# GARANTE QUE O ARRAY EH COMPLEXO
		x = np.asarray( arr, dtype=np.complex )
		
		# PREPARA OS VALORES DA SOMATORIA
		N = arr.shape[0]
		n = np.arange( N )
		k = n.reshape( ( N, 1 ) )
		M = np.exp( -2j * np.pi * k * n / N )
		
		# RETORNA A MULTIPLICACAO FINAL
		return np.dot( M, x )

def idft( arr_fourrier ):
	if( isinstance( arr_fourrier, np.ndarray ) ):
		# GARANTE QUE O ARRAY EH COMPLEXO
		x = np.asarray( arr_fourrier, dtype=np.complex )
		
		# PREPARA OS VALORES DA SOMATORIA
		N = arr_fourrier.shape[0]
		n = np.arange( N )
		k = n.reshape( ( N, 1 ) )
		M = np.exp( 2j * np.pi * k * n / N )
		
		# RETORNA A MULTIPLICACAO FINAL
		return np.dot( M, x )

def dft_img( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_fourrier = np.asarray( img, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = dft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = dft( img_fourrier[:, j] )
			
			return img_fourrier.astype( np.uint8 )
		elif( nchannels( img ) == 3 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_fourrier = np.asarray( img, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = dft_img( img[:, :, canal] )
			return img_fourrier.astype( np.uint8 )

def idft_img( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_fourrier = np.asarray( img, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = idft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = idft( img_fourrier[:, j] )
			
			return img_fourrier.astype( np.uint8 )
		elif( nchannels( img ) == 3 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_fourrier = np.asarray( img )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = idft_img( img[:, :, canal] )
			return img_fourrier

def fft( arr ):
	if( isinstance( arr, np.ndarray ) ):
		x = np.asarray( arr, dtype=np.complex )
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
		arr_fourrier = np.asarray( arr, dtype=np.complex )
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
		arr_fourrier = np.asarray( arr_fourrier, dtype=np.complex )

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
		arr = np.asarray( arr_fourrier, dtype=np.complex )
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
path = listaImagens[1]

img = imread( path )
print( img.shape )

img_fft_np = np.asarray( img, dtype=np.uint8 )
for c in range( img.shape[2] ):
	temp = np.fft.fft2( img[:,:,c] )
	img_fft_np[:,:,c] = temp.astype( np.uint8 )

#img_fft_np = np.fft.fft2( img ).astype( np.uint8 )
img_dft = dft_img( img )

print( img_fft_np )
print( img_dft )
print( "Igualdade:", np.allclose( img_fft_np, img_dft ) )

"""
img1 = imread( path )
img2 = fft_2d( img1 )
img3 = np.fft.fft2( img1 )

ifft_img1 = ifft_2d( img1 )
ifft_img2 = ifft_2d( img2 )
ifft_img3 = ifft_2d( img3 )

ifft_img1_np = np.fft.ifft2( img1 )
ifft_img2_np = np.fft.ifft2( img2 )
ifft_img3_np = np.fft.ifft2( img3 )

img1 = img1.astype( np.uint8 )
img2 = img2.astype( np.uint8 )
img3 = img3.astype( np.uint8 )

ifft_img1 = ifft_img1.astype( np.uint8 )
ifft_img2 = ifft_img2.astype( np.uint8 )
ifft_img3 = ifft_img3.astype( np.uint8 )

ifft_img1_np = ifft_img1_np.astype( np.uint8 )
ifft_img2_np = ifft_img2_np.astype( np.uint8 )
ifft_img3_np = ifft_img3_np.astype( np.uint8 )

compare_images( img2, img3, "FFT Meu", "FFT Numpy", "Transformada de Fourrier" )
compare_images( ifft_img1, ifft_img1_np, "iFFT Meu", "iFFT Numpy", "Transformada inversa de Fourrier" )
compare_images( ifft_img2, ifft_img2_np, "iFFT Meu", "iFFT Numpy", "Transformada inversa de Fourrier" )
compare_images( ifft_img3, ifft_img3_np, "iFFT Meu", "iFFT Numpy", "Transformada inversa de Fourrier" )
"""