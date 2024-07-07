# Master Thesis : Active Learning for Semi-Supervised image classification

## Abstract

Machine learning models are algorithms highly dependent on the amount
and quality of data they are given. In real world scenarios, it’s difficult to
get quality and quantity of labeled data. Nevertheless, unlabeled data is
much easier to obtain and with a suitable usage, it could help to improve
the models. In this sense, Active Learning and Semi-Supervised learning
focus on how to use efficiently the unlabeled data. Active Learning aims to
find the unlabeled examples that would improve the most the model when
given labeled for training, while Semi-supervised learning focuses on using
unlabeled data to increase the model performance in an unsupervised manner.

In this Master Thesis we aim to combine ideas from both settings in order
to obtain a more general way to train the image classifiers and to label new
instances. In particular, we will use a pre-trained encoder to improve the
classifier accuracy and study the different Active Learning strategies with
the idea of generating a surrogate model that is able to predict the expected
increase in performance after labeling a given instance.
