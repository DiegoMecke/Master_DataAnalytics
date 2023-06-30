#Import packages
import numpy as np

#Define data generator
def loadData(batch_size=100, n_perturbed=0):
    while True:
        x = np.random.rand(batch_size,1)*2*np.pi-np.pi 
        y = np.sin(x)
        y[np.random.randint(0,batch_size,n_perturbed)]=1000
        yield x , y