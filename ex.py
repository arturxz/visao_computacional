
import numpy as np 
import matplotlib.pyplot as plt 

from math import floor

def calculoLacoMascara( mask ):
    mas_max_lin, mas_max_col = mask.shape
    
    mas_hal_lin = floor( mas_max_lin / 2 )
    mas_hal_col = floor( mas_max_col / 2 )

    print( "mas_hal_lin", mas_hal_lin)
    print( "mas_hal_col", mas_hal_col)

    for i in range( mas_max_lin ):
        for j in range( mas_max_col ):
            print( "i:", i, "j:", j )
            print( "idx i:", i-mas_hal_lin, "idx j:", j-mas_hal_col )
            print( mask[ i, j ] )


mask = np.ones( (2, 2), np.uint8 ) * 1/9
calculoLacoMascara( mask )