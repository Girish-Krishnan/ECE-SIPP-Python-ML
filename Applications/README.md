# Example Applications

The `Applications` folder contains standalone demos that showcase popular computer vision libraries. Each script can be run individually with Python.

Make sure you install the required packages before running the examples. All of them can be installed with `pip` as shown below.

```bash
pip install mediapipe ultralytics segment-anything diffusers
```

### 0. Pose estimation with Mediapipe
`python 00_pose_estimation_mediapipe.py`

Opens your webcam and draws human pose landmarks on each frame. Press **q** to quit.

### 1. Face mesh with Mediapipe
`python 01_face_mesh_mediapipe.py`

Streams the webcam and overlays a detailed facial mesh. Press **q** to exit.

### 2. YOLOv8 object detection
`python 02_yolov8_webcam.py`

Runs the lightweight YOLOv8 model from the `ultralytics` package on your webcam stream. The model weights are automatically downloaded the first time you run it.

### 3. Segment Anything demo
`python 03_segment_anything.py`

Loads `sample.jpg` from the current folder and segments the central region using Meta's Segment Anything model. A mask is drawn in green on the image before it is displayed.

### 4. Text-to-image diffusion
`python 04_text_to_image_diffusion.py`

Prompts you for a text description and uses a Stable Diffusion pipeline to generate an image saved as `generated.png`.
