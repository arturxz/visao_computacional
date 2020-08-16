import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

# ARQUIVOS LOSS SEGMENTACAO
loss_bcdu1 = "csv_saidas/loss/loss_BCDU_net_D1.csv"
loss_bcdu3 = "csv_saidas/loss/loss_BCDU_net_D3.csv"
loss_unvgg16 = "csv_saidas/loss/loss_unet_vgg16.csv"
loss_unet  = "csv_saidas/loss/loss_unet_cnn.csv"

# ARQUIVOS IoU SEGMENTACAO
acc_bcdu1 = "csv_saidas/acc/meaniou_BCDU_net_D1.csv"
acc_bcdu3 = "csv_saidas/acc/meaniou_BCDU_net_D3.csv"
acc_unvgg16 = "csv_saidas/acc/meaniou_unet_vgg16.csv"
acc_unet  = "csv_saidas/acc/meaniou_unet_cnn.csv"

# ARQUIVOS LOSS CLASSIFICACAO
loss_inceptionResNetV2  = "csv_saidas/class/loss_inceptionResNetV2.csv"
loss_inceptionv3        = "csv_saidas/class/loss_inceptionv3.csv"
loss_mobilenet          = "csv_saidas/class/loss_mobilenet.csv"
loss_mobilenetv2        = "csv_saidas/class/loss_mobilenetv2.csv"
loss_ResNet101V2        = "csv_saidas/class/loss_ResNet101V2.csv"
loss_vgg16              = "csv_saidas/class/loss_vgg16.csv"
loss_vgg19              = "csv_saidas/class/loss_vgg19.csv"

# ARQUIVOS ACCURACY CLASSIFICACAO
acc_inceptionResNetV2   = "csv_saidas/class/acc_inceptionResNetV2.csv"
acc_inceptionv3         = "csv_saidas/class/acc_inceptionv3.csv"
acc_mobilenet           = "csv_saidas/class/acc_mobilenet.csv"
acc_mobilenetv2         = "csv_saidas/class/acc_mobilenetv2.csv"
acc_ResNet101V2         = "csv_saidas/class/acc_ResNet101V2.csv"
acc_vgg16               = "csv_saidas/class/acc_vgg16.csv"
acc_vgg19               = "csv_saidas/class/acc_vgg19.csv"

def plot_graph_7_line( x1, y1, l1, x2, y2, l2, x3, y3, l3, x4, y4, l4, x5, y5, l5, x6, y6, l6, x7, y7, l7, title_graph="Comparação", x_label="Epochs", y_label="Valor"):
    linha1, = plt.plot( x1, y1, '-', label=l1 )
    linha1.set_antialiased(True)

    linha2, = plt.plot( x2, y2, '-', label=l2 )
    linha2.set_antialiased(True)

    linha3, = plt.plot( x3, y3, '-', label=l3 )
    linha3.set_antialiased(True)

    linha4, = plt.plot( x4, y4, '-', label=l4 )
    linha4.set_antialiased(True)

    linha5, = plt.plot( x5, y5, '-', label=l5 )
    linha5.set_antialiased(True)

    linha6, = plt.plot( x6, y6, '-', label=l6 )
    linha6.set_antialiased(True)

    linha7, = plt.plot( x7, y7, '-', label=l7 )
    linha7.set_antialiased(True)

    plt.yscale('log')
    plt.title( title_graph )
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    plt.legend(loc='upper left')
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
    plt.legend(loc='upper_right')
    #plt.grid(True)
    plt.show()

def plot_graph_1_line( x_axis, y_axis, title_graph="Gráfico", x_label="Epochs", y_label="Valor", color="b" ):
    linha1, = plt.plot( x_axis, y_axis, '-', color=color)
    linha1.set_antialiased(True)

    plt.title( title_graph )
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    #plt.legend(loc='upper right')
    #plt.grid(True)
    plt.show()

def plot_from_csv( path_file, cnn_name, val_name, color="b" ):

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
        
        #plot_graph_1_line( step, value, cnn_name, "Epochs", val_name, color )
        return step, value

""" 
# SEGMENTACAO
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
"""

# CLASSIFICACAO
step_loss_inceptionResNetV2, val_loss_inceptionResNetV2 = plot_from_csv( loss_inceptionResNetV2, "Inception-ResNetv2", "Loss", "r" )
step_loss_inceptionv3, val_loss_inceptionv3             = plot_from_csv( loss_inceptionv3, "InceptionV3", "Loss", "r" )
step_loss_mobilenet, val_loss_mobilenet                 = plot_from_csv( loss_mobilenet, "MobileNet", "Loss", "r" )
step_loss_mobilenetv2, val_loss_mobilenetv2             = plot_from_csv( loss_mobilenetv2, "MobileNetV2", "Loss", "r" )
step_loss_ResNet101V2, val_loss_ResNet101V2             = plot_from_csv( loss_ResNet101V2, "ResNet101V2", "Loss", "r" )
step_loss_vgg16, val_loss_vgg16                         = plot_from_csv( loss_vgg16, "VGG16", "Loss", "r" )
step_loss_vgg19, val_loss_vgg19                         = plot_from_csv( loss_vgg19, "VGG19", "Loss", "r" )

step_acc_inceptionResNetV2, val_acc_inceptionResNetV2   = plot_from_csv( acc_inceptionResNetV2, "Inception-ResNetv2", "Accuracy", "b" )
step_acc_inceptionv3, val_acc_inceptionv3               = plot_from_csv( acc_inceptionv3, "InceptionV3", "Accuracy", "b" )
step_acc_mobilenet, val_acc_mobilenet                   = plot_from_csv( acc_mobilenet, "MobileNet", "Accuracy", "b" )
step_acc_mobilenetv2, val_acc_mobilenetv2               = plot_from_csv( acc_mobilenetv2, "MobileNetV2", "Accuracy", "b" )
step_acc_ResNet101V2, val_acc_ResNet101V2               = plot_from_csv( acc_ResNet101V2, "ResNet101V2", "Accuracy", "b" )
step_acc_vgg16, val_acc_vgg16                           = plot_from_csv( acc_vgg16, "VGG16", "Accuracy", "b" )
step_acc_vgg19, val_acc_vgg19                           = plot_from_csv( acc_vgg19, "VGG19", "Accuracy", "b" )

plot_graph_7_line(  step_loss_inceptionResNetV2, val_loss_inceptionResNetV2, "Inception-ResNetv2",
                    step_loss_inceptionv3, val_loss_inceptionv3, "InceptionV3",
                    step_loss_mobilenet, val_loss_mobilenet, "MobileNet",
                    step_loss_mobilenetv2, val_loss_mobilenetv2, "MobileNetV2",
                    step_loss_ResNet101V2, val_loss_ResNet101V2, "ResNet101V2",
                    step_loss_vgg16, val_loss_vgg16, "VGG16",
                    step_loss_vgg19, val_loss_vgg19, "VGG19",
                    "Comparação Loss", "Epochs", "Loss" )

plot_graph_7_line(  step_acc_inceptionResNetV2, val_acc_inceptionResNetV2, "Inception-ResNetv2",
                    step_acc_inceptionv3, val_acc_inceptionv3, "InceptionV3",
                    step_acc_mobilenet, val_acc_mobilenet, "MobileNet",
                    step_acc_mobilenetv2, val_acc_mobilenetv2, "MobileNetV2",
                    step_acc_ResNet101V2, val_acc_ResNet101V2, "ResNet101V2",
                    step_acc_vgg16, val_acc_vgg16, "VGG16",
                    step_acc_vgg19, val_acc_vgg19, "VGG19",
                    "Comparação Accuracy", "Epochs", "Accuracy" )
