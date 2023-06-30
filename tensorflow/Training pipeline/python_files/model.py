#Import packages
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

#Define model class
class MyModel(Model):
    def __init__(self, neurons=[12,12,3], reg=0.001, activation="relu"):
        super(MyModel, self).__init__()

        self.denseLayers=[]
        for idx,neuron in enumerate(neurons):
            self.denseLayers.append(Dense(neuron, kernel_regularizer=regularizers.l2(reg), activation=activation)) #Update

        self.outputLayer = Dense(1, kernel_regularizer=regularizers.l2(reg), activation=None)

    def call(self, input_x):
        output = input_x

        for layer in self.denseLayers:
            output = layer(output)

        return self.outputLayer(output)