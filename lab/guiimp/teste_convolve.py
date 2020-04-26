import os
import platform
import threading

import time as t
import numpy as np
import matplotlib.pyplot as mp

from PIL import Image
from math import floor

def imread( filename ):
	img = mp.imread( filename, np.uint8 )
	return img

def imshow( img ):
	plt = mp.imshow(img, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
	plt.set_interpolation('nearest')
	mp.show()

def is_gray_image( img ):
	if( isinstance(img, np.ndarray) ):
		if( img.ndim == 2 ):
			return True
		else:
			return False
	else:
		return False

def maskBlur():
	print("maskBlur")
	mask = np.array( ([ [1, 2, 1], [2, 4, 2], [1, 2, 1] ]) ) * 1/16
	return mask

def calcula_pixel_p_b( img, img_i, img_j, mask, half_mask, convoluted_img ):    
    # PIXEL ATUAL
	pxl_atual = 0
	convoluted_window = np.zeros( (mask.shape[0],mask.shape[1]), np.float32 )
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

path_p_b = "C:\\Users\\asnascimento\\Documents\\Mestrado\\visao_computacional\\lab\\guiimp\\img\\cinza_3.jpg"
path_cor = "C:\\Users\\asnascimento\\Documents\\Mestrado\\visao_computacional\\lab\\guiimp\\img\\cor_6.tiff"

img_p_b = imread( path_p_b )
img_cor = imread( path_cor )

print( "-------------------------------------")
ini = t.time()
img_res_p_b = convolve( img_p_b, maskBlur() )
print( "normal  :", t.time() - ini )

print( "-------------------------------------")
ini = t.time()
img_res_cor = convolve( img_cor, maskBlur() )
print( "normal  :", t.time() - ini )

imshow( img_p_b )
imshow( img_res_p_b )
imshow( img_cor )
imshow( img_res_cor )