# PyTorch Deep Learning Tutorials

There are two main ways to run the examples in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/3_Deep_Learning/deep_learning_tutorials.ipynb)

The notebook mirrors the scripts: datasets are split into training, validation and test sets and the training progress is visualized with loss curves.

## Option 2: Python Scripts

You can also execute the scripts individually with `python`. Make sure PyTorch and torchvision are installed:

```bash
pip install torch torchvision
```

### 0. PyTorch tensor basics
`python 00_tensor_basics.py`

Demonstrates creating tensors, running operations, and basic autograd.

### 1. Train a simple MLP
`python 01_train_mlp.py`

Downloads the MNIST dataset, splits it into training/validation/test sets and trains a multilayer perceptron for a few epochs. A plot of the learning curve is saved along with the weights as `mlp_mnist.pt`.

### 2. Train a basic CNN
`python 02_train_cnn.py`

Builds a small convolutional neural network trained on the same MNIST splits. Training and validation loss curves are plotted and the final weights are written to `cnn_mnist.pt`.

### 3. Finetune a pretrained ResNet
`python 03_transfer_learning.py`

Uses a ResNet18 pretrained on ImageNet and finetunes it on CIFAR10 using proper train/val/test splits. Learning curves are saved and the finetuned weights are written to `resnet18_cifar10.pt`.

### 4. Run object detection
`python 04_object_detection.py`

Loads a pretrained Faster R-CNN model and performs object detection on your own image (save as `sample.jpg`). Detected boxes with high confidence are displayed using Matplotlib.

### 5. Autoencoder for MNIST
`python 05_autoencoder.py`

Trains a simple autoencoder on MNIST with validation and test evaluation. A loss curve is generated and the trained weights are saved as `autoencoder_mnist.pt`.

### 6. Inference with a saved CNN
`python 06_inference_example.py`

Loads the CNN from script 02 and runs inference on a sample from the test set.

These short scripts provide a starting point for experimenting with deep learning in computer vision. Feel free to modify the code or plug in your own datasets to learn more!
