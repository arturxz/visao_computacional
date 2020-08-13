import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1,51)
amostra1 = np.random.uniform(low=0.0, high=100.0, size=(50,))
amostra2 = np.random.uniform(low=0.0, high=100.0, size=(50,))
amostra3 = np.random.uniform(low=0.0, high=100.0, size=(50,))

linhas = {
    'Amostra 1' : amostra1,
    'Amostra 2' : amostra2,
    'Amostra 3' : amostra3
}

#"""
linha1, = plt.plot( epochs, amostra1, '-', label="Amostra 1" )
linha1.set_antialiased(True)

linha2, = plt.plot( epochs, amostra2, '-', label="Amostra 2" )
linha2.set_antialiased(True)

linha3, = plt.plot( epochs, amostra3, '-', label="Amostra 3" )
linha3.set_antialiased(True)
#"""

#plt.plot( linhas )

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(loc='upper right')
#plt.grid(True)
plt.show()