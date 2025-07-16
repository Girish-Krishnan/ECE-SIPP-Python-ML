# Machine Learning Tutorial

This repository contains the Jupyter notebook `Machine_Learning_Tutorial.ipynb` which introduces machine learning concepts using Python and NumPy. The tutorial walks through implementations of models such as linear regression, logistic regression, k-NN, decision trees and k-Means. It now also includes a worked example on the `sklearn` digits dataset (a smaller version of MNIST) so you can practice on real-world data.

You can open the notebook locally or in [Google Colab](https://colab.research.google.com/) to run the examples interactively.

## Opening in Google Colab

1. Navigate to the notebook file on GitHub.
2. Replace `github.com` in the URL with `colab.research.google.com/github`.
   - Example: `https://colab.research.google.com/github/<username>/<repo>/blob/main/Machine_Learning_Tutorial.ipynb`
3. Press Enter and Colab will load the notebook so you can execute the code online.

Colab provides free GPU/CPU resources, making it a convenient way to experiment with the code without installing anything locally.

---

## OpenCV Webcam Tutorials

The `OpenCV_Tutorials` folder contains a series of small Python scripts that demonstrate fun things you can do with a webcam. Each file builds on the previous one, so run them in order to get a feel for OpenCV's capabilities. Make sure you have `opencv-python` installed in your environment.

### 0. Show your webcam
`python OpenCV_Tutorials/00_show_webcam.py`

Opens the webcam and simply streams the video feed. Press **q** to quit.

### 1. Grayscale filter
`python OpenCV_Tutorials/01_gray_filter.py`

Converts every frame to grayscale so you can see how easy it is to manipulate the image data.

### 2. Edge detection
`python OpenCV_Tutorials/02_edge_detection.py`

Applies a Canny edge detector on the grayscale frames to highlight edges in real time.

### 3. Face detection
`python OpenCV_Tutorials/03_face_detection.py`

Draws rectangles around detected faces using a built-in Haar cascade. Try it with multiple people in the frame!

### 4. Cartoon effect
`python OpenCV_Tutorials/04_cartoonize.py`

Transforms the webcam feed into a cartoon-style image using bilateral filtering and adaptive thresholding.

### 5. Background subtraction
`python OpenCV_Tutorials/05_background_subtraction.py`

Segments moving objects from the background, a common first step in many vision applications.

Each script can be stopped by focusing the video window and pressing **q**. Feel free to tweak the parameters in the code and experiment further. Have fun exploring OpenCV!
