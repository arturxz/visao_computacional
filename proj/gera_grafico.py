import matplotlib.pyplot as plt
import numpy as np

def retorna_vetores_unet_256_256():
    arr_acc = np.flip( np.array([0.6161,0.5903,0.5744,0.5596,0.4297,0.3542,0.3245,0.3174,0.2826,0.2751,0.2575,0.2453,0.2583,0.2663,0.2327,0.2239,0.2444,0.2152,0.2024,0.2095,0.2045,0.1821,0.1735,0.1684,0.1556,0.1464,0.1539,0.1569,0.1457,0.1524,0.1393,0.1291,0.1179,0.1117,0.1072,0.0990,0.0903,0.0904,0.0871,0.0846,0.0750,0.0719,0.0722,0.0662,0.0601,0.0515,0.0546]) ) * 100
    arr_lss = np.flip( np.array([0.3660,0.3660,0.3660,0.3660,0.3660,0.3660,0.3662,0.3660,0.3660,0.3660,0.3660,0.3660,0.3669,0.3660,0.3660,0.3660,0.3663,0.3660,0.3686,0.3660,0.3661,0.3663,0.3661,0.3666,0.3662,0.3663,0.3700,0.3661,0.3675,0.3667,0.3725,0.3699,0.3737,0.3751,0.3789,0.3886,0.3988,0.3907,0.3984,0.4089,0.4130,0.4193,0.4075,0.4233,0.4227,0.4543,0.4566]) ) * 100
    return arr_acc, arr_lss

print("Executando")

epochs = np.arange(1,51)
amostra1 = np.random.uniform([36.6,36.6,36.52,36.54,36.6,36.62,])
amostra2 = np.random.uniform(low=0.0, high=100.0, size=(50,))
amostra3 = np.random.uniform(low=0.0, high=100.0, size=(50,))

"""
linha1, = plt.plot( epochs, amostra1, '-', label="Amostra 1" )
linha1.set_antialiased(True)

linha2, = plt.plot( epochs, amostra2, '-', label="Amostra 2" )
linha2.set_antialiased(True)

"""

accu, loss = retorna_vetores_unet_256_256()
linha1, = plt.plot( np.arange(1,accu.size+1), accu, '-', label="IoU" )
linha1.set_antialiased(True)

linha2, = plt.plot( np.arange(1,loss.size+1), loss, '--', label="Loss" )
linha2.set_antialiased(True)

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(loc='upper right')
#plt.grid(True)
plt.show()

"""

"""