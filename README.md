# rotnet

Rotational equivariant network using vector fields.

## Overview

Image channels are divided into blocks of 3.  The first channel is a scalar field, and the second two are the components of vector fields.

The convolution layer includes kernels of a specific form to be rotationally equivariant, including a specific form for maps between two scalar fields, maps between two vector fields, and maps between scalar and vector fields.

Additionally, to maintain rotational equivariance, we have a new nonlinearity that depends only on vector (and scalar) magnitude, a new batchnorm layer, and a new downsampling layer.

We include implementations of a ResNet18 network (used in the MEDMNIST paper) and a ResNet20 network (used in the original resnet paper when applied to the CIFAR10 dataset).   We also include an implementation of a new version of these, we call RotNet18 and RotNet20.

