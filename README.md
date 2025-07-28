# Rotnet

To follow along with Dr. Tward's presentation at this year's CGSI, you can find the demo [here](https://github.com/twardlab/rotnet).

## Overview
The principle of translation equivariance (if an input image is translated, then an output image should be translated by the same amount), led to the development of convolutional neural networks (CNNs) that revolutionized machine vision. Other symmetries, like rotations and reflections, play a similarly critical role, especially in biomedical image analysis. However, exploiting these symmetries has not seen wide adoption. We hypothesize that this is partially due to the mathematical complexity of methods used to exploit these symmetries, which often rely on representation theory, a bespoke concept in differential geometry and group theory. In this work, we show that the same equivariance can be achieved using a simple form of convolution kernels that we call “**moment kernels**,” and prove that all equivariant kernels must take this form. These are a set of radially symmetric functions of a spatial position *x*, multiplied by powers of the components of _x_ or the identity matrix. We implement equivariant neural networks using standard convolution modules, and provide architectures to execute several biomedical image analysis tasks that depend on equivariance principles: _classification_ (outputs are invariant under orthogonal transforms), _3D image registration_ (outputs transform like a vector), and _cell segmentation_ (quadratic forms defining ellipses transform like a matrix).

## Moment Kernel Definitions
Below are the definitions of the moment kernels for maps between scalar-valued functions, vector-valued functions, and tensor-valued functions where the tensor can be of any rank. 
1. Maps between two scalar fields
   - $k(x) = f_{ss}(|x|)$
2. Maps between scalar and vector fields
   - $k(x) = f_{sv}(|x|)x$
3. Maps between vector and scalar fields
   - $k(x) = f_{vs}(|x|)x^{T}$
4. Maps between two vector fields
   - $k(x) = f_{vv0}(|x|)id + f_{vv1}(|x|)xx^{T}$
5. Maps between 2 general tensors of rank _r_
   - $k^{i_{1},...,i_{r}}(x) = f_{\emptyset}(|x|)x^{i_{1},...,i_{r}}$

## Moment Kernel Implementation
We define a radial function along one axis with a fixed number of samples (here 3), and resample it into a hybercube (here 3 × 3, or 3 × 3 × 3) for convolution using linear interpolation. Interpolation weights are precomputed and executed as matrix multiplication.

When a convolution module is initialized, all signatures for a given rank _r_ tensor are enumerated, and one such radial function is randomly initialized for each. The kernel is constructed as a rank _r_ array, and reshaped using lexicographic ordering to give the correct input and output dimensions ($i × d$ and $(r − i) × d$ using our previous notation).

Multiple such kernels of different ranks are stacked into a single matrix-valued kernel, the blocks of which map between tensor fields of different rank, which is used in a standard convolution layer. Input and output tensor-valued images of different rank are also stacked using lexicographic ordering. This leads to a linear acting on a set of tensor-valued features being applied in one standard convolution operation. Due to discretization, our networks are exactly equivariant for 90-degree rotations and reflections and only approximately equivariant for other angles.

## Experimental Results
1. Image Classification
!(/images/classification_results.png)
2. Image Registration
!(/images/registration_results.png)
3. Cell Detection
!(/images/detection_results.png)
# Further Reading
If you are interested in reading more details about our derivations and implementations related to this work, you can find a preliminary draft of our paper [here](https://arxiv.org/abs/2505.21736).
