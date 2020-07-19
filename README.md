# Machine Learning Experiments

This repository is a collection of the various machine learning experiments I've undertaken while learning deep learning fundamentals.

The majority of these experiments so far have their roots in the [Dive into Deep Learning](https://d2l.ai) guide, which I have found extremely benficial.

The first 5 experiments are directly taken from the first 7 chapters of the guide. Experiment 6 is my attempt at solving a limited version of the Google Quick Draw dataset.

## 6. Google Quick Draw CNN

The Google Quick Draw dataset contains simple doodles of an extremely wide variety of objects. The dataset is available in various formats: 
1. the original data, containing stroke and timing details as the drawer draws the doodle
2. A simplified version of the above, with timing information removed.
3. Pre-rendered images in Numpy data files, containing tens of thousands of 28x28 images per class.

I have chosen to use a CNN for classification of the pre-rendered images (the first two formats are more suitable for RNN's, as in Google's SketchRNN).

To keep the scope limited and trainable on my home computer, I have used 5000 training images in each of the first 51 classes of objects.

The model chosen is a slight variation of the ResNet-18 architecture. Early training on the ResNet-18 architecture revealed underfitting on the 5,000 image dataset. I therefore decided to greatly widen the Residual blocks.

Model summary:
```
Model: "ResNet-18"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              multiple                  3200      
_________________________________________________________________
batch_normalization (BatchNo multiple                  256       
_________________________________________________________________
activation (Activation)      multiple                  0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) multiple                  0         
_________________________________________________________________
resnet_block (ResnetBlock)   multiple                  526976    
_________________________________________________________________
resnet_block_1 (ResnetBlock) multiple                  2102528   
_________________________________________________________________
resnet_block_2 (ResnetBlock) multiple                  8399360   
_________________________________________________________________
resnet_block_3 (ResnetBlock) multiple                  33575936  
_________________________________________________________________
global_average_pooling2d (Gl multiple                  0         
_________________________________________________________________
dense (Dense)                multiple                  52275     
=================================================================
Total params: 44,660,531
Trainable params: 44,645,043
Non-trainable params: 15,488
_________________________________________________________________
None
```

After training for 10 epochs, the model was able to acheive 95% accuracy on the test set of 1000 images. Here is a sample output of one 128 image batch:

![Prediction Output](./outputs/predictions-2020-07-18T08:46:20.373203.png)

The training accuract after 10 epochs was similar to the test accuracy. Future experiments could explore a more complex model, as the data does not seem to overfit.