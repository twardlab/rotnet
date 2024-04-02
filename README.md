# rotnet

Rotational equivariant network using vector fields.

## Overview

Image channels are divided into blocks of 3.  The first channel is a scalar field, and the second two are the components of vector fields.

The convolution layer includes kernels of a specific form to be rotationally equivariant, including a specific form for maps between two scalar fields, maps between two vector fields, and maps between scalar and vector fields.

Additionally, to maintain rotational equivariance, we have a new nonlinearity that depends only on vector (and scalar) magnitude, a new batchnorm layer, and a new downsampling layer.

We include implementations of a ResNet18 network (used in the MEDMNIST paper) and a ResNet20 network (used in the original resnet paper when applied to the CIFAR10 dataset).   We also include an implementation of a new version of these, we call RotNet18 and RotNet20.



## ToDo for paper
We have two versions of the new conv layer.  One version that is rotation invariant, and one that is rotation and reflection invariant.  A will call them rotnet and refnet because these terms sound something like resnet.

We have two "standard" resnet architectures.  A 20 layer version used in the resnet paper for CIFAR 10, and an 18 layer version used in the medmnist paper.

For these networks feature maps are grouped into blocks of three.  The first represents a scalar field, and the second two represent a vector field.

For each of these we want to test 3 versions of my rotnets and 3 versions of my resnets.  

* About same number of channels at the first layer (but less) than corresponding resnet
* About the same number of channels as the first layer (but less) than the coresponding resnet, counting the two vector components as one channel.
* About the same number of parameters (but less) than the corresponding resnet.

Daniel has done all these for the rotnet, but only looked at one for the refnet.  Daniel also did an extra rotnet with 5x5 kernels instead of 3x3 which I think we can remove.

For each of these networks  (and the two vanilla resnets) I want to train and evaluate them on each of the MEDMNIST 2D datasets. I want to repeat the evaluation 3 times.  Make sure to save the evalution results (but not necessarily the network parameters which could be too big to save a million times).  

How many networks will we test in each case? 2 vanilla resnets, 2 rotnets with equal features, 2 rotnets with equal independenet features, 2 rotnets with equal parameters, 2 refnets with equal features, 2 refnets with equal independent features, 2 refnets with equal parameters.  

This means that for each dataset we will train and evaluate 14 networks, and do this 3 times.  There are 12 2D medmnist datasets, but only 11 are binary multi-classification. 

When we report quality, we'll want to consider which of these datasets we actually beleive to be rotationally invariant.

I think it is Blood, Path, Derma, Path, maybe Tissue.

I hypothesize that our architecture will do better on the rotationally invariant datasets, but not on the others.  I hypothesize that our architecture will do better with a smaller number of samples and may not matter when we have a lot of samples.

## Other interesting thigs to look at
It would be nice to compare this to resnets with data augmentation, even though we are claiming this could replace augmentation.

It would be nice, for demonstration purposes, to evaluate the resnets on a rotated version of the training set.

Ultimately we will want to implement this in 3D.


