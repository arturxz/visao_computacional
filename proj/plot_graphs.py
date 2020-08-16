import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os

# ARQUIVOS LOSS
loss_bcdu1 = "csv_saidas/loss/loss_BCDU_net_D1.csv"
loss_bcdu3 = "csv_saidas/loss/loss_BCDU_net_D3.csv"
loss_vgg16 = "csv_saidas/loss/loss_unet_vgg16.csv"
loss_unet  = "csv_saidas/loss/loss_unet_cnn.csv"

# ARQUIVOS LOSS
acc_bcdu1 = "csv_saidas/acc/meaniou_BCDU_net_D1.csv"
acc_bcdu3 = "csv_saidas/acc/meaniou_BCDU_net_D3.csv"
acc_vgg16 = "csv_saidas/acc/meaniou_unet_vgg16.csv"
acc_unet  = "csv_saidas/acc/meaniou_unet_cnn.csv"

def plot_graph_1_line( x_axis, y_axis, title_graph="Gráfico", x_label="Epochs", y_label="Valor" ):
    linha1, = plt.plot( x_axis, y_axis, '-' )
    linha1.set_antialiased(True)

    plt.title( title_graph )
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    #plt.legend(loc='upper right')
    #plt.grid(True)
    plt.show()

def plot_graph_4_line( x1, y1, l1, x2, y2, l2, x3, y3, l3, x4, y4, l4, title_graph="Comparação", x_label="Epochs", y_label="Valor"):
    linha1, = plt.plot( x1, y1, '-', label=l1 )
    linha1.set_antialiased(True)

    linha1, = plt.plot( x2, y2, '-', label=l2 )
    linha1.set_antialiased(True)

    linha1, = plt.plot( x3, y3, '-', label=l3 )
    linha1.set_antialiased(True)

    linha1, = plt.plot( x4, y4, '-', label=l4 )
    linha1.set_antialiased(True)

    plt.title( title_graph )
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    plt.legend(loc='upper center')
    #plt.grid(True)
    plt.show()

def plot_from_csv( path_file, cnn_name, val_name ):

    with open( path_file ) as csv_file:
        # ABRINDO CSV
        csv_reader = csv.reader( csv_file, delimiter="," )

        step = []
        value = []

        # POPULANDO DICTIONARYS
        for row in csv_reader:
            if( row[1].lower() != "step" ):
                step.append( int(row[1]) )
                value.append( float(row[2]) )
        
        plot_graph_1_line( step, value, cnn_name, "Epochs", val_name )
        return step, value

step_loss_unet, val_loss_unet = plot_from_csv( loss_unet, "U-Net", "Loss" )
step_loss_vgg16, val_loss_vgg16 = plot_from_csv( loss_vgg16, "UNet-VGG16", "Loss" )
step_loss_bcdu1, val_loss_bcdu1 = plot_from_csv( loss_bcdu1, "UNet-Resnet", "Loss")
step_loss_bcdu3, val_loss_bcdu3 = plot_from_csv( loss_bcdu3, "DC-UNet", "Loss")

step_acc_unet, val_acc_unet = plot_from_csv( acc_unet, "U-Net", "Mean IoU" )
step_acc_vgg16, val_acc_vgg16 = plot_from_csv( acc_vgg16, "UNet-VGG16", "Mean IoU" )
step_acc_bcdu1, val_acc_bcdu1 = plot_from_csv( acc_bcdu1, "UNet-Resnet", "Mean IoU" )
step_acc_bcdu3, val_acc_bcdu3 = plot_from_csv( acc_bcdu3, "DC-UNet", "Mean IoU" )

plot_graph_4_line(  step_loss_unet, val_loss_unet,   "U-Net",
                    step_loss_vgg16, val_loss_vgg16, "UNet-VGG16",
                    step_loss_bcdu1, val_loss_bcdu1, "UNet-Resnet",
                    step_loss_bcdu3, val_loss_bcdu3, "DC-UNet",
                    "Comparação Loss", "Epochs", "Loss" )

plot_graph_4_line(  step_acc_unet, val_acc_unet,   "U-Net",
                    step_acc_vgg16, val_acc_vgg16, "UNet-VGG16",
                    step_acc_bcdu1, val_acc_bcdu1, "UNet-Resnet",
                    step_acc_bcdu3, val_acc_bcdu3, "DC-UNet",
                    "Comparação Mean IoU", "Epochs", "Mean IoU" )