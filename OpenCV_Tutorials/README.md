# OpenCV Webcam Tutorials

The `OpenCV_Tutorials` folder contains a series of small Python scripts that demonstrate fun things you can do with a webcam. Each file builds on the previous one, so run them in order to get a feel for OpenCV's capabilities. Make sure you have `opencv-python` installed in your environment.

### 0. Show your webcam
`python 00_show_webcam.py`

Opens the webcam and simply streams the video feed. Press **q** to quit.

### 1. Grayscale filter
`python 01_gray_filter.py`

Converts every frame to grayscale so you can see how easy it is to manipulate the image data.

### 2. Edge detection
`python 02_edge_detection.py`

Applies a Canny edge detector on the grayscale frames to highlight edges in real time.

### 3. Face detection
`python 03_face_detection.py`

Draws rectangles around detected faces using a built-in Haar cascade. Try it with multiple people in the frame!

### 4. Cartoon effect
`python 04_cartoonize.py`

Transforms the webcam feed into a cartoon-style image using bilateral filtering and adaptive thresholding.

### 5. Background subtraction
`python 05_background_subtraction.py`

Segments moving objects from the background, a common first step in many vision applications.

### 6. Record video to file
`python 06_video_writer.py`

Captures frames from the webcam and saves them to `output.avi` while displaying the stream.

### 7. Draw shapes and text
`python 07_draw_shapes.py`

Creates a blank canvas and draws rectangles, circles, lines and text to demonstrate OpenCV's drawing functions.

Each script can be stopped by focusing the video window and pressing **q**. Feel free to tweak the parameters in the code and experiment further. Have fun exploring OpenCV!
