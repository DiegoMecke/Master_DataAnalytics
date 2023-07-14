# Introduction to Tensorflow : Set training pipeline
### By: Diego Coello de Portugal Mecke

This notebook aims to create model trainning pipeline without using standard tensorflow functionalities.

The data is synthetic (sinuosoidal function) with some outliers for test robustness of the model.
The training pipeline will be tested with a dataloader and a generator to prove the generality of the implemented functionalities.

The usage of regularization and gradient clipping will be tested for the case with outliers, showing a comparison of the performance of the different settings.

Lastly, the code will be rewritten in .py files and loaded to prove the usability for large scale projects.
