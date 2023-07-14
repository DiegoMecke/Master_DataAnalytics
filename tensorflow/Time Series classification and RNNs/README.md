# Recurrent Neural Networks
### By: Diego Coello de Portugal Mecke

This notebook aims to experiment with RNN (Recurrent Neural Networks) in the Time Series domain. The dataset used is [PeekDB](https://www.scitepress.org/Papers/2018/65852/65852.pdf), in particular, the data has been obtained from https://github.com/RafaelDrumond/PeekDB. This dataset reports the body position and orientation when doing specific activities (Crouching, Running, Swing, etc.). The architectures used will be [LSTMS and GRU's](https://asset-pdf.scinapse.io/prod/2944851425/2944851425.pdf).

The notebook will: 

 - Normalize the data and write a data generator which outputs data pairs of shape x=(Batchsize,60,20), y=(Batchsize,1). The generator should have an argument "mode" which specifies whether we want train/test. Use Actor 0 and 1 for testing, 2 for val, and actor 3 and onwards for training. Limit the experiment to classes "Crouching", "Running", "Swing".
 - Build a LSTM/GRU architecture to succesfully classify this dataset for the classes "Crouching", "Running" and "Swing". Actor 2 will be used for validation, while actor 3 and onwards will be used for training. Additionaly, early stopping will be used on the validation set.
 - Do some experiments regarding the window size will be done.
 - Do some data augmentation to try to improve the model performance.
