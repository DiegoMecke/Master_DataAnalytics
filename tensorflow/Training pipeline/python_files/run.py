#Import packages
from python_files.train import MyTrainModel
from python_files.data_loading import loadData
from python_files.model import MyModel
import numpy as np
import matplotlib.pyplot as plt


def run_experiment():
    
    #Define hyperparameters
    reg = 0.01
    data_size = 5500
    batch_size = 8
    lr = 0.001
    epochs = 30
    n_perturbed = 1
    
    #Load dataloader
    train_gen = loadData(data_size, n_perturbed)
    test_gen  = loadData(data_size)
    
    #Initialize model and optimizer
    model = MyModel(reg=reg)
    opt = MyTrainModel(model, batch_size=batch_size, lr=lr)
    
    #Run experiment
    hist = opt.run(train_gen, test_gen, epochs, verbose=1)
    
    #Show results
    fig, axs = plt.subplots(1,3,figsize=(15,5))

    #Train loss histogram
    axs[0].plot(np.mean(hist,-1)[:,0],label="train")
    axs[0].set_title("Training Loss Convergence")
    axs.flat[0].set(xlabel='Epochs', ylabel='MSE')
    
    #Test loss histogram
    axs[1].plot(np.mean(hist,-1)[:,1],label="test")
    axs[1].set_title("Test Loss Convergence")
    axs.flat[1].set(xlabel='Epochs', ylabel='MSE')
    
    #Model prediction
    x_test, y_test = next(loadData(1000))
    axs[2].scatter(x_test,y_test,label="true")
    axs[2].scatter(x_test, model(x_test),label="pred")