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

def imread_gray( filename ):
	img = mp.imread( filename, np.uint8 )
	if( is_gray_image( img ) ):
		return img
	else:
		return rgb2gray( img )

def imshow( img ):
	plt = mp.imshow(img, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt.set_interpolation('nearest')
	mp.show()

def nchannels( img ):
	if( isinstance( img, np.ndarray ) ):
		if( img.ndim == 2 ):
			return 1
		else:
			return img.shape[2]
	else:
		return -1

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

def size( img ):
	if( isinstance( img, np.ndarray ) ):
		return (img.shape[1], img.shape[0])
	else:
		return None

def is_gray_image( img ):
	if( isinstance(img, np.ndarray) ):
		if( img.ndim == 2 ):
			return True
		else:
			return False
	else:
		return False

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

def negative( img ):
	print("method: negative")
	if( isinstance( img, np.ndarray ) ):
		return  255 - img
	return None

def contrast( img, r, m ):
	print("method: contrast")
	print("r:", r, "m:", m)
	if( isinstance( img, np.ndarray ) ):
		if( is_gray_image( img ) ):
			img_contrast = np.zeros( img.shape, np.uint64 )
			img_contrast = ( m +( r * ( img - m ) ) )
			img_contrast[ img_contrast > 255 ] = 255
			return np.asarray( img_contrast, np.uint8 )
		else:
			img_contrast_r = np.zeros( (img.shape[0], img.shape[1]), np.uint64 )
			img_contrast_g = np.zeros( (img.shape[0], img.shape[1]), np.uint64 )
			img_contrast_b = np.zeros( (img.shape[0], img.shape[1]), np.uint64 )

			img_contrast_r = ( m +( r * ( img[:, :, 0] - m ) ) )
			img_contrast_g = ( m +( r * ( img[:, :, 1] - m ) ) )
			img_contrast_b = ( m +( r * ( img[:, :, 2] - m ) ) )
			
			img_contrast = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), np.uint64 )
			
			img_contrast[:, :, 0] = img_contrast_r
			img_contrast[:, :, 1] = img_contrast_g
			img_contrast[:, :, 2] = img_contrast_b

			img_contrast[ img_contrast > 255 ] = 255
			
			return np.asarray( img_contrast, np.uint8 )

	return None

def hist( img ):
	print( "method hist" )
	if( isinstance(img, np.ndarray) ):
		if( is_gray_image( img ) ):
			return np.reshape( np.bincount( img.ravel(), minlength=256 ), (256, 1) )
		else:
			hist = np.zeros( (256, 3), np.uint32 )
			hist[:, 0] = np.reshape( np.bincount( img[:, :, 0].ravel(), minlength=256 ), (256) ).astype( np.uint32 )
			hist[:, 1] = np.reshape( np.bincount( img[:, :, 1].ravel(), minlength=256 ), (256) ).astype( np.uint32 )
			hist[:, 2] = np.reshape( np.bincount( img[:, :, 2].ravel(), minlength=256 ), (256) ).astype( np.uint32 )
			
			return hist
			
	return None

def showhist( col_hist, bins=1 ):
	print( "method showhist" )
	print( "bins:", bins )
	if( isinstance(col_hist, np.ndarray) ):
		print( col_hist.shape )
		if( col_hist.shape == (256,0) or col_hist.shape == (256, 1) ):
			lista = []
			for i in range( 0, 255 ):
				if( col_hist[i, 0] > 0 ):
					for j in range( 0, col_hist[i, 0] ):
						lista.append( i )
			
			mp.hist( lista, bins=bins, color='k', alpha=0.75 )
			mp.title( "Histograma de imagem em escala de cinza" )
			mp.xlabel( "Intensidades" )
			mp.ylabel( "Quantidade de ocorrências" )
			mp.show()
		else:
			lista_r = []
			lista_g = []
			lista_b = []
			
			for i in range( 0, 255 ):
				if( col_hist[i, 0] > 0 ):
					for j in range( 0, col_hist[i, 0] ):
						lista_r.append( i )
				if( col_hist[i, 1] > 0 ):
					for j in range( 0, col_hist[i, 1] ):
						lista_g.append( i )
				if( col_hist[i, 2] > 0 ):
					for j in range( 0, col_hist[i, 2] ):
						lista_b.append( i )
			
			mp.hist(lista_r, bins=bins, color='r', alpha=0.75, label='Red')
			mp.hist(lista_g, bins=bins, color='g', alpha=0.75, label='Green')
			mp.hist(lista_b, bins=bins, color='b', alpha=0.75, label='Blue')			
			mp.title( "Histograma de imagem colorida RGB" )
			mp.xlabel( "Intensidades de cor" )
			mp.ylabel( "Quantidade de ocorrências" )
			mp.show()

def histeq( img ):
	print( "method histeq" )
	if( isinstance(img, np.ndarray) ):
		if( not is_gray_image( img ) ):
			img = rgb2gray( img )
		
		# CALCULANDO CDF
		cdf = np.cumsum( hist( img ).flatten() )
		
		# MASCARANDO PIXELS 0
		cdf_m = np.ma.masked_equal( cdf, 0 )

		# CALCULA HISTOGRAMA ACUMULADO [T(rk)]
		cdf_m = ( ( cdf_m - cdf_m.min() ) * 255 ) / ( cdf_m.max() - cdf_m.min() )
		
		# NORMALIZA HISTOGRAMA [0, 255]
		cdf_f = np.ma.filled( cdf_m, 0 ).astype( 'uint8' )
		
		# FAZ A TRANSFORMACAO DA IMAGEM
		img_ret = cdf_f[img]
		
		return img_ret

def calcula_pixel_p_b( img, img_i, img_j, mask, half_mask, convoluted_img ):    
    # PIXEL ATUAL
	pxl_atual = 0
	for mask_i in range( mask.shape[0] ):
		for mask_j in range( mask.shape[1] ):
			# CALCULANDO INDICES DA IMAGEM
			idx_img_i = img_i + ( mask_i - half_mask[0] )
			idx_img_j = img_j + ( mask_j - half_mask[1] )

			# VERIFICANDO LIMIARES DA IMAGEM
			if( idx_img_i >= img.shape[0] ):
				idx_img_i = img.shape[0] - 1
			elif( idx_img_i < 0 ):
				idx_img_i = 0
							
			if( idx_img_j >= img.shape[1] ):
				idx_img_j = img.shape[1] - 1
			elif( idx_img_j < 0 ):
				idx_img_j = 0

			# EXECUTANDO SOMA DA MASCARA NO PIXEL ATUAL
			pxl_atual = pxl_atual + img[idx_img_i, idx_img_j] * mask[mask_i, mask_j]
					
	# ATRIBUINDO NOVA COR AO PIXEL ATUAL
	pxl_atual = floor( pxl_atual )
	if( pxl_atual > 255 ):
		pxl_atual = 255
	elif( pxl_atual < 0 ):
		pxl_atual = 0
	convoluted_img[img_i, img_j] = pxl_atual

def calcula_pixel_cor( img, img_i, img_j, mask, half_mask, convoluted_img ):
	# PIXEL ATUAL
	pxl_atual_r = 0
	pxl_atual_g = 0
	pxl_atual_b = 0

	for mask_i in range( mask.shape[0] ):
		for mask_j in range( mask.shape[1] ):
			# CALCULANDO INDICES DA IMAGEM
			idx_img_i = img_i + ( mask_i - half_mask[0] )
			idx_img_j = img_j + ( mask_j - half_mask[1] )
			# VERIFICANDO LIMIARES DA IMAGEM
			if( idx_img_i >= img.shape[0] ):
				idx_img_i = img.shape[0] - 1
			elif( idx_img_i < 0 ):
				idx_img_i = 0
							
			if( idx_img_j >= img.shape[1] ):
				idx_img_j = img.shape[1] - 1
			elif( idx_img_j < 0 ):
				idx_img_j = 0

			# EXECUTANDO SOMA DA MASCARA NO PIXEL ATUAL
			pxl_atual_r = pxl_atual_r + img[idx_img_i, idx_img_j, 0] * mask[mask_i, mask_j]
			pxl_atual_g = pxl_atual_g + img[idx_img_i, idx_img_j, 1] * mask[mask_i, mask_j]
			pxl_atual_b = pxl_atual_b + img[idx_img_i, idx_img_j, 2] * mask[mask_i, mask_j]
					
	# ATRIBUINDO NOVA COR AO PIXEL ATUAL
	pxl_atual_r = floor( pxl_atual_r )
	if( pxl_atual_r > 255 ):
		pxl_atual_r = 255
	elif( pxl_atual_r < 0 ):
		pxl_atual_r = 0
					
	pxl_atual_g = floor( pxl_atual_g )
	if( pxl_atual_g > 255 ):
		pxl_atual_g = 255
	elif( pxl_atual_g < 0 ):
		pxl_atual_g = 0
					
	pxl_atual_b = floor( pxl_atual_b )
	if( pxl_atual_b > 255 ):
		pxl_atual_b = 255
	elif( pxl_atual_b < 0 ):
		pxl_atual_b = 0
	
	convoluted_img[img_i, img_j, 0] = pxl_atual_r
	convoluted_img[img_i, img_j, 1] = pxl_atual_g
	convoluted_img[img_i, img_j, 2] = pxl_atual_b


def convolve( img, mask ):
	print( "convolve method" )
	print( "img shape:", img.shape)
	print( "mask shape:", mask.shape)

	half_mask = floor( mask.shape[0] / 2 ), floor( mask.shape[1] / 2 )

	if( isinstance( img, np.ndarray ) and isinstance( mask, np.ndarray ) ):
		convoluted_img = np.zeros( img.shape, np.uint8 )

		if( is_gray_image( img ) ):
			for img_i in range( img.shape[0] ):
				for img_j in range( img.shape[1] ):
					calcula_pixel_p_b( img, img_i, img_j, mask, half_mask, convoluted_img )                    
		else:
			for img_i in range( img.shape[0] ):
				for img_j in range( img.shape[1] ):
					calcula_pixel_cor( img, img_i, img_j, mask, half_mask, convoluted_img )

		return convoluted_img
	return None

def maskBlur():
	print("maskBlur")
	mask = np.array( ([ [1, 2, 1], [2, 4, 2], [1, 2, 1] ]) ) * 1/16
	return mask

def blur( img ):
	print("blur")
	return convolve( img, maskBlur() )

def blur_2( img ):
	print("blur 2")
	img_blured = convolve( img, (maskBlur()[1,:] * 2).reshape(1,3) )
	img_blured = convolve( img_blured, (maskBlur()[1,:] * 2).reshape(3,1) )
	return img_blured

def seSquare3():
	print("maskBlur")
	mask = np.array( ([ [255, 255, 255], [255, 255, 255], [255, 255, 255] ]) )
	return mask

def seCross3():
	print("seCross3")
	mask = np.array( ([ [0, 255, 0], [255, 255, 255], [0, 255, 0] ]) )
	return mask

def morph_convolve( img, strEl, morph ):
	print( "img:", img.shape )
	print( "str:", strEl.shape )
	
	if( isinstance( img, np.ndarray ) and isinstance( strEl, np.ndarray ) ):
		half_strel_i = floor( strEl.shape[0] / 2 )
		half_strel_j = floor( strEl.shape[1] / 2 )

		img_final = np.copy( img )

		if( is_gray_image( img ) ):
			for i in range( img.shape[0] ):
				for j in range( img.shape[1] ):
					if( morph == 'e' ): #erosion
						controle = 255
					elif( morph == 'd' ): #dilatation
						controle = 0
					for k in range( -half_strel_i, half_strel_i+1):
						for l in range( -half_strel_j, half_strel_j+1 ):
							if( strEl[k+half_strel_i, l+half_strel_j] != 0 ):
								idx_img_i = i+k
								idx_img_j = j+l

								if( idx_img_i >= img.shape[0] ):
									idx_img_i = img.shape[0] - 1
								elif( idx_img_i < 0 ):
									idx_img_i = 0
												
								if( idx_img_j >= img.shape[1] ):
									idx_img_j = img.shape[1] - 1
								elif( idx_img_j < 0 ):
									idx_img_j = 0

								if( morph == 'e' ): #erosion
									if( img[idx_img_i, idx_img_j] < controle ):
										controle = img[idx_img_i, idx_img_j]
								elif( morph == 'd' ): #dilatation
									if( img[idx_img_i, idx_img_j] > controle ):
										controle = img[idx_img_i, idx_img_j]
					img_final[i,j] = controle
		else:
			for i in range( img.shape[0] ):
				for j in range( img.shape[1] ):
					if( morph == 'e' ): #erosion
						controle_r = 255
						controle_g = 255
						controle_b = 255
					elif( morph == 'd' ): #dilatation
						controle_r = 0
						controle_g = 0
						controle_b = 0
					for k in range( -half_strel_i, half_strel_i+1):
						for l in range( -half_strel_j, half_strel_j+1 ):
							if( strEl[k+half_strel_i, l+half_strel_j] != 0 ):
								idx_img_i = i+k
								idx_img_j = j+l

								if( idx_img_i >= img.shape[0] ):
									idx_img_i = img.shape[0] - 1
								elif( idx_img_i < 0 ):
									idx_img_i = 0
												
								if( idx_img_j >= img.shape[1] ):
									idx_img_j = img.shape[1] - 1
								elif( idx_img_j < 0 ):
									idx_img_j = 0

								if( morph == 'e' ): #erosion
									if( img[idx_img_i, idx_img_j, 0] < controle_r ):
										controle_r = img[idx_img_i, idx_img_j, 0]
									if( img[idx_img_i, idx_img_j, 1] < controle_g ):
										controle_g = img[idx_img_i, idx_img_j, 1]
									if( img[idx_img_i, idx_img_j, 2] < controle_b ):
										controle_b = img[idx_img_i, idx_img_j, 2]
								elif( morph == 'd' ): #dilatation
									if( img[idx_img_i, idx_img_j, 0] > controle_r ):
										controle_r = img[idx_img_i, idx_img_j, 0]
									if( img[idx_img_i, idx_img_j, 1] > controle_g ):
										controle_g = img[idx_img_i, idx_img_j, 1]
									if( img[idx_img_i, idx_img_j, 2] > controle_b ):
										controle_b = img[idx_img_i, idx_img_j, 2]
					
					img_final[i,j,0] = controle_r
					img_final[i,j,1] = controle_g
					img_final[i,j,2] = controle_b
		
	return img_final

def erode( img, strEl ):
	print("erode")
	return morph_convolve(img, strEl, 'e')
	
def dilate( img, strEl ):
	print("convolve")
	return morph_convolve(img, strEl, 'd')

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

def fft( arr ):
	if( isinstance( arr, np.ndarray ) ):
		x = np.asarray( arr, dtype=np.complex )
		N = x.shape[0]

		if( N % 2 > 0 ):
			# FAZER PREENCHIMENTO DA IMAGEM (AINDA NAO FAZ)
			print( arr.shape )
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

def ifft( arr_fourrier ):
	if( isinstance( arr_fourrier, np.ndarray ) ):
		# GARANTINDO QUE O ARRAY EH DE COMPLEXOS
		arr = np.asarray( arr_fourrier, dtype=np.complex )

		# RETORNA A CONJUGAÇÃO DA PARTE IMAGINARIA
		arr_fourrier_conjugate = np.conjugate( arr_fourrier )
		
		# CALCULA O FOURRIER DO ARRAY CONJUGADO
		arr = fft( arr_fourrier_conjugate )
		
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

def fft_2d( arr ):
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

def ifft_2d( arr_fourrier ):
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
		
		return arr

def fft_2d_np( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			# ENTAO E IMAGEM EH ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = np.fft.fft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = np.fft.fft( img_fourrier[:, j] )
			
			return img_fourrier
		elif( nchannels( img ) > 1 ):
			# ENTAO PARA CADA CANAL DE COR, CHAMA A SI MESMO
			# COMO SE FOSSE ESCALA DE CINZA
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = fft_2d_np( img[:, :, canal] )
			return img_fourrier

def ifft_2d_np( img ):
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
				img_inv_fourrier[:, :, canal] = ifft_2d_np( img[:, :, canal] )
			return img_inv_fourrier

"""
    MAIN AQUI
"""

listaImagens = retornaListaArquivos( "C:\\Users\\artur\\Google Drive\\UFS\\Mestrado\\[2020.1] Periodo 1\\COMPU0026 - VISAO COMPUTACIONAL\\lab\\guiimp\\img_fft" )
path = listaImagens[1]

img = imread( path )
print( "shape da imagem:", img.shape, "\n" )

ini = t.time()
img_fft = fft_2d( img )
print( "Tempo Execução Fourrier FFT:", t.time()-ini )

ini = t.time()
img_ifft = ifft_2d( img_fft )
print( "Tempo Execução Fourrier Inversa iFFT:", t.time()-ini )

print( "ori, fft:", np.allclose( img, img_ifft ) )
print( "fft, ift:", np.allclose( img_fft, img_ifft ) )

img_fft = np.real( img_fft ).astype( np.uint8 )
img_ifft = np.real( img_ifft ).astype( np.uint8 )

compare_images( img, img_fft, "Imagem Original", "Imagem Fourrier" )
compare_images( img_fft, img_ifft, "Imagem Fourrier", "Imagem Inversa Fourrier" )

"""
ini = t.time()
img_idft = idft_2d( img_dft )
print( "Tempo Execução Inversa Fourrier:", t.time()-ini )

ini = t.time()
img_idft_np = idft_2d_np( img_dft )
print( "Tempo Execução Inversa Fourrier Numpy:", t.time()-ini )

print( "Fourrier Inversa tá ok?", np.allclose( img_idft_np, img_idft ), "\n" )

img_dft = img_dft.astype( np.uint8 )
img_idft = img_idft.astype( np.uint8 )
img_idft_np = img_idft_np.astype( np.uint8 )

compare_images( img, img_dft, "Imagem Original", "Imagem Fourrier" )
compare_images( img_dft, img_idft, "Imagem Fourrier", "Imagem Inversa Fourrier" )
compare_images( img_idft, img_idft_np, "Minha", "Numpy" )
"""







"""
	---------------------------------
	EXEMPLOS DE COMO USAR OS METODOS
	---------------------------------

from skimage import morphology as morph
retornaListaArquivos()
listaImagens = retornaListaArquivos()

img = imread( listaImagens[8] )

#imshow( contrast( img, 1.2, 1.1 ) )

imshow( rgb2gray( img ) )

for img in listaImagens:
    print( img, size( img ) )

imshow( rgb2gray( img ) )

imshow( imreadgray( listaImagens[0] ) )

imshow( thresh( img, 150 ) )

imshow( negative( img ) )

#img_res = convolve( img, maskBlur() )

img_th = thresh( img, 240 )

img_ero = erode( img, seCross3() )

img_dil = dilate( img_th, seCross3() )


compare_images( img_ero, img, "Erode", "Erode Referencia" )
compare_images( img_dil, img, "Dilate", "Dilate Referencia" )

"""