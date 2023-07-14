# Image segmentation with U-Net
### By: Diego Coello de Portugal Mecke

This notebook aims to:

1) Reimplement [U-Net architecture](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf) by Ronneberge et al. (2015) for semantic image segmentation. Meaning an encoder-decoder sequence-to-sequence model with skip connections between each layer of the encoder and decoder. There will be some experimentation with a  pretrained encoder (e.g. MobileNetV2).

2) An Augment class and the function **iou** which computes a per-class Intersection-over-Union score will be implemented. The augment class should increase the model performance while the **iou** metric should be a better way to review the model performance.

The trainning will be stopped at 20 iterations due to time constrains. Different architectures and training procedures will be compared.

Lastly, there will be a short experiment on finetuning.

Note: The dataset was to heavy to upload it on github. In the notebook you will find a cell commented which can be runned on Google Colab to obtain the dataset and run the project locally.
