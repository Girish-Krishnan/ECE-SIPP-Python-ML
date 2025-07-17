# Python Machine Learning Examples

This repository collects small but complete examples for learning scientific computing and machine learning with Python. The code is organized by topic so you can jump directly to the areas that interest you.

## Directory overview
- **Machine_Learning_Tutorial.ipynb** – introductory Jupyter notebook covering classic algorithms implemented with NumPy.
- **NumPy_Tutorials/** – standalone scripts illustrating common NumPy operations.
- **OpenCV_Tutorials/** – webcam and image processing demos using OpenCV.
- **Deep_Learning_Tutorials/** – PyTorch based training scripts from simple MLPs to transfer learning.
- **Applications/** – fun demos using Mediapipe, YOLO, Segment Anything and more.

See the individual `README.md` files inside each folder for details on the available examples.

## Installation and setup
1. Install **Python 3.8** or newer. We recommend using a virtual environment:
   ```bash
   python3 -m venv ml-env
   source ml-env/bin/activate
   ```
2. Install the required packages with `pip`:
   ```bash
   pip install numpy matplotlib opencv-python torch torchvision
   ```
   Additional packages such as `imageio` may be needed for some examples. They can be installed on demand using `pip install <package>`.
3. Clone this repository and navigate into it:
   ```bash
   git clone <repo-url>
   cd ECE-SIPP-Python-ML
   ```

## Running the tutorials
Every script can be executed directly with Python. For example, to run one of the NumPy demos:
```bash
python NumPy_Tutorials/01_array_math.py
```

The deep learning scripts will automatically download datasets like MNIST the first time they are run. Training epochs are kept short so they finish quickly on a CPU.

## Opening the notebook in Google Colab
1. Navigate to the notebook file on GitHub.
2. Replace `github.com` in the URL with `colab.research.google.com/github`.
   - Example: `https://colab.research.google.com/github/<username>/<repo>/blob/main/Machine_Learning_Tutorial.ipynb`
3. Press Enter and Colab will load the notebook so you can execute the code online.

Colab provides free GPU/CPU resources, making it a convenient way to experiment with the code without installing anything locally.
