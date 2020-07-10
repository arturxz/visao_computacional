# BIBLIOTECAS BASICAS
import csv
import cv2
import numpy as np

# PARA MOSTRAR IMAGENS
import matplotlib.pyplot as mp

# PARA IA
import tensorflow as tf
from tensorflow import keras
from skimage import io, transform

# PARA DATA
from datetime import datetime

model = create_model()
model.load_weights( "modelos_salvos/generico/test_24-06-2020_10-06-43" )
print( model )