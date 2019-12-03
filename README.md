# FaceRecognitionModule
Two models to recognize faces images from the [CMU Faces Images Dataset](http://archive.ics.uci.edu/ml/datasets/cmu+face+images).

One model (**face_recon_short_arc.py**) is a shallow CNN net, but it obtained low loss and high accuracy in this dataset with just a few epochs! It is a good model to use for not too complex problems and situations of data acquisition.

The second model (**face_recon_transfer_learning.py**) applies the transfer learning technique to use the trained weights of ResNet-50 on the ImageNet dataset to classify these face images.

In order to do that we freeze the feature extraction layers of the ResNet-50 (i.e. make the weights not trainable) and add on top of these feature extraction layers a sequence of dense layers (i.e. fully connected layers) to learn how to classify the faces given the extracted features.

The architecture of these predictive models are exported to **json** files and the weights are exported to **h5** files, that can be used in many applications.

One application that uses these model is in [this repository](https://github.com/muriloHoracio/imobiliariaWeb).
