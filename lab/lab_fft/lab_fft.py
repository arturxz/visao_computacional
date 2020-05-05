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
	if( nchannels( img ) > 3 ):
		return img[:, :, 0:3]
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

def compare_images( img1, img2, sub1="Imagem 1", sub2="Imagem 2", titulo="Comparando Imagens" ):
	mp.subplot( 1, 2, 1 )

	mp.suptitle( titulo )

	plt1 = mp.imshow(img1, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt1.set_interpolation('nearest')
	mp.title( sub1 )
	
	mp.subplot( 1, 2, 2 )
	plt2 = mp.imshow(img2, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt2.set_interpolation('nearest')
	mp.title( sub2 )

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

def idft( arr ):
	if( isinstance( arr, np.ndarray ) ):
		# GARANTE QUE O ARRAY EH COMPLEXO
		x = np.asarray( arr, dtype=np.complex )
		
		# PREPARA OS VALORES DA SOMATORIA
		N = arr.shape[0]
		n = np.arange( N )
		k = n.reshape( ( N, 1 ) )
		M = np.exp( 2j * np.pi * k * n / N )
		
		# RETORNA A MULTIPLICACAO FINAL
		return 1 / N * np.dot( M, x )

def maior_potencia_2_mais_proxima( n ):
	i = 0
	while( n != 0 ):
		n = n >> 1
		i = i + 1
	
	return 1 << i

def eh_potencia_de_2( n ):
	return n and not( n & (n-1) )

def fft( arr ):
	arr = np.asarray( arr, dtype=np.complex )
	N = arr.shape[0]

	if( not eh_potencia_de_2( N ) ):
		# FAZER PREENCHIMENTO DA IMAGEM (AINDA NAO FAZ)
		print( arr.shape )
		raise ValueError( "Precisa ser potência de 2" )
	
	if( N <= 16 ):
		return dft( arr ) # CASO BASE DA RECURSAO
	else:
		metade_pares = fft( arr[::2] ) # COMPUTA PRIMEIRA METADE
		metade_impar = fft( arr[1::2] ) # COMPUTA SEGUNDA METADE
		exponencial = np.exp( -2j * np.pi * np.arange( N ) / N ) # PREPARA O EXPONENCIAL

		fu = metade_pares + metade_impar * exponencial[ :int(N//2) ]
		fu_k = metade_pares + metade_impar * exponencial[ int(N//2): ]

		return np.concatenate( [ fu, fu_k ] )

def ifft( arr_fourrier ):
	# GARANTINDO QUE O ARRAY SEJA DE COMPLEXOS
	arr = np.asarray( arr_fourrier, dtype=np.complex )

	# RETORNA A CONJUGAÇÃO DA PARTE IMAGINARIA
	arr = np.conjugate( arr_fourrier )
		
	# CALCULA O FOURRIER DO ARRAY CONJUGADO
	arr = fft( arr )
		
	# RECONJUGANDO
	arr = np.conjugate( arr )

	# DIVISAO POR M
	arr = arr / arr_fourrier.shape[0]
	return arr

def dft_2d( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = dft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = dft( img_fourrier[:, j] )
			
			return img_fourrier
		elif( nchannels( img ) > 1 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = dft_2d( img[:, :, canal] )
			return img_fourrier

def idft_2d( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_inv_fourrier = np.zeros( img.shape, dtype=np.complex )
			for j in range( img.shape[1] ):
				img_inv_fourrier[:, j] = np.fft.ifft( img[:, j] )

			for i in range( img.shape[0] ):
				img_inv_fourrier[i, :] = np.fft.ifft( img_inv_fourrier[i, :] )
			
			return img_inv_fourrier
		elif( nchannels( img ) > 1 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_inv_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_inv_fourrier[:, :, canal] = idft_2d( img[:, :, canal] )
			return img_inv_fourrier

def fft_2d( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = fft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = fft( img_fourrier[:, j] )
			
			return img_fourrier
		elif( nchannels( img ) > 1 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = fft_2d( img[:, :, canal] )
			return img_fourrier


def ifft_2d( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = ifft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = ifft( img_fourrier[:, j] )
			
			return img_fourrier
		elif( nchannels( img ) > 1 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = ifft_2d( img[:, :, canal] )
			return img_fourrier

""" 
	Main
"""
listaImagens = retornaListaArquivos()
"""imagem = listaImagens[1]
img = imread( imagem )

print( imagem )
print( img.shape )

imshow( img )

# EXECUTANDO FFT
ini = t.time()
img_fft = fft_2d( img )
print( "Tempo Execução FFT:", t.time()-ini )

# EXECUTANDO iFFT
ini = t.time()
img_ifft = ifft_2d( img_fft )
print( "Tempo Execução iFFT:", t.time()-ini, "\n" )

# CONVERTENDO PARA QUE SEJAM VISIVEIS COMO IMAGENS
img_fft = np.real( img_fft ).astype( np.uint8 )
img_ifft = np.real( img_ifft ).astype( np.uint8 )

compare_images( img_fft, img_ifft, "Imagem FFT", "Imagem iFFT", imagem.split('\\').pop() )"""

for imagem in listaImagens:
	img = imread( imagem )
	print( imagem )
	print( img.shape )

	imshow( img )

	# EXECUTANDO FFT
	ini = t.time()
	img_fft = fft_2d( img )
	print( "Tempo Execução FFT:", t.time()-ini )

	# EXECUTANDO iFFT
	ini = t.time()
	img_ifft = ifft_2d( img_fft )
	print( "Tempo Execução iFFT:", t.time()-ini, "\n" )

	# CONVERTENDO PARA QUE SEJAM VISIVEIS COMO IMAGENS
	img_fft = np.real( img_fft ).astype( np.uint8 )
	img_ifft = np.real( img_ifft ).astype( np.uint8 )

	compare_images( img_fft, img_ifft, "Imagem FFT", "Imagem iFFT", imagem.split('\\').pop() )