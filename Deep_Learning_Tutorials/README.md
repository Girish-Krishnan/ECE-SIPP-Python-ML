# PyTorch Deep Learning Tutorials

The `Deep_Learning_Tutorials` folder walks through training neural networks with PyTorch. Run the scripts in order to get a quick introduction to different model types. Each example can be executed with `python` followed by the file name.

### 0. Train a simple MLP on MNIST
`python 00_train_mlp.py`

Downloads the MNIST dataset and trains a small multilayer perceptron for one epoch. The resulting weights are saved to `mlp_mnist.pt`.

### 1. Train a basic CNN
`python 01_train_cnn.py`

Builds a small convolutional neural network and trains it on MNIST. After one epoch the model is stored as `cnn_mnist.pt`.

### 2. Finetune a pretrained ResNet
`python 02_transfer_learning.py`

Uses a ResNet18 pretrained on ImageNet and finetunes it on CIFAR10. This demonstrates how to adapt larger models for a new task. The finetuned weights are written to `resnet18_cifar10.pt`.

### 3. Run object detection
`python 03_object_detection.py`

Loads a pretrained Faster R-CNN model and performs object detection on your own image (save as `sample.jpg`). Detected boxes with high confidence are displayed using Matplotlib.

These short scripts provide a starting point for experimenting with deep learning in computer vision. Feel free to modify the code or plug in your own datasets to learn more!
