
# Image Classification with CNNs and Transfer Learning

## Overview

This project explores image classification with PyTorch using the CIFAR-10 dataset. It implements a convolutional neural network (CNN), introduces data augmentation and dropout to address overfitting, and applies transfer learning with a pretrained ResNet-18 model.

The notebook covers image preprocessing, CNN architecture, model training, validation, learning curves, regularization, and transfer learning.

## Dataset

- **Name:** CIFAR-10
- **Source:** [University of Toronto — CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Access:** Downloaded using `torchvision.datasets.CIFAR10`
- **Total images:** 60,000
- **Image dimensions:** 32 × 32 pixels
- **Image format:** RGB
- **Number of classes:** 10
- **Training set:** 50,000 images
- **Test/validation set:** 10,000 images

The ten classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Each image belongs to one of these classes.

## Workflow

### 1. Image preprocessing and visualization

The notebook downloads CIFAR-10 and creates PyTorch `DataLoader` objects with a batch size of 64.

The initial preprocessing pipeline:

- Converts images to tensors using `transforms.ToTensor()`.
- Normalizes the RGB channels using a mean of 0.5 and a standard deviation of 0.5 for each channel.
- Displays example images and their class labels.

### 2. Building a CNN from scratch

The first model, `conv_net`, contains two convolutional blocks:

- **First block:** convolution with 16 output channels and a 5 × 5 kernel, followed by `Tanh` and max pooling.
- **Second block:** convolution with 128 output channels and a 5 × 5 kernel, followed by `Tanh` and max pooling.

The classification block contains:

- A fully connected layer with 64 hidden units and ReLU activation.
- An output layer with 10 units, one for each CIFAR-10 class.

The model is trained using:

- **Loss:** Cross-entropy loss
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning rate:** 0.01

The notebook implements a training loop and plots training and validation loss and accuracy.

### 3. Data augmentation

To introduce variation into the training images, the notebook applies:

- Random horizontal flips with probability 0.2.
- Random rotations between −10° and +10°.
- Tensor conversion and channel normalization.

The validation images receive tensor conversion and normalization, without random augmentation.

### 4. Dropout regularization

A second CNN, `conv_net_dropout`, uses the same general two-block convolutional structure and adds a dropout layer with `p=0.2` after the first fully connected layer's ReLU activation.

The model is trained using the augmented training dataset. Learning curves are plotted to examine the behavior of training and validation loss and accuracy.

### 5. Transfer learning with ResNet-18

The final section uses a pretrained ResNet-18 model from `torchvision.models`.

The notebook:

- Loads pretrained ResNet-18 weights.
- Freezes the parameters of the pretrained network.
- Replaces the final fully connected layer with a new layer containing 10 output units for CIFAR-10.
- Trains the new classification layer using SGD with a learning rate of 0.01.

The notebook reports **5,130 trainable parameters** after freezing the pretrained network and replacing its final layer.

Training and validation curves are plotted to examine the model's behavior.

## Technologies

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib

## Evaluation notes

The notebook includes a validation function intended to calculate loss and accuracy on a supplied data loader. However, the function iterates over `train_loader` instead of the supplied `loader` and uses `.cuda()` directly. Consequently, the reported validation metrics do not reliably measure performance on the intended validation dataset, and the function may fail in a CPU-only environment.

For this reason, the notebook's reported curves and metrics should not be treated as a reliable comparison of the models without correcting the validation function and rerunning the experiments.
