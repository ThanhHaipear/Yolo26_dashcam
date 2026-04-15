# YOLO26 Dashcam Detection

Real-time object detection demo for dashcam and webcam streams using Ultralytics YOLO and OpenCV.

This project runs inference directly from a camera feed, draws detection boxes on each frame, and overlays FPS to demonstrate real-time performance. It is a compact computer vision demo suitable for a portfolio, internship application, or CV submission.

## Project Overview

The goal of this project is to build a simple but practical real-time detection pipeline for camera input. The application:

- loads a trained YOLO model from a local `.pt` file
- reads frames from a webcam or dashcam source
- performs object detection on each frame
- renders bounding boxes and labels on the output
- displays live FPS for runtime monitoring

## Demo

Add your own media files into `assets/demo/`, then update the paths below.

### Demo GIF

![YOLO Dashcam Demo](assets/demo/demo.gif)

### Demo Screenshot

![Detection Result](assets/demo/demo-frame.jpg)

### Demo Video

GitHub README does not play local video files inline reliably. Use one of these options:

1. Upload a short demo video to LinkedIn, YouTube, Google Drive, or GitHub Release and paste the public link here.
2. Convert a short screen recording to `demo.gif` and place it in `assets/demo/`.
3. Add 2 to 4 screenshots that show different detection results.

Example:

```md
[Watch demo video](https://your-link-here)
```

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch model weights (`.pt`)

## Repository Structure

```text
demo1/
|- assets/
|  \- demo/
|     \- .gitkeep
|- camhanhtrinh.py
|- supbest.pt
|- README.md
|- .gitignore
```

## Main File

The main application entry point is:

```text
camhanhtrinh.py
```

Core behavior in the script:

- resolves the model path from the current project directory
- opens the camera stream using OpenCV
- runs YOLO inference on each frame
- visualizes predictions with `result.plot()`
- calculates and displays FPS

## How To Run

### 1. Install dependencies

```bash
pip install ultralytics opencv-python
```

If your environment does not already include PyTorch, install it first based on your CUDA or CPU setup.

### 2. Prepare the model

Make sure the model file exists in the project root:

```text
supbest.pt
```

### 3. Start the application

```bash
python camhanhtrinh.py
```

## Configuration

You can edit these values in `camhanhtrinh.py`:

- `MODEL_PATH`: path to the trained YOLO model
- `CAMERA_INDEX`: webcam index, usually `0` or `1`
- `CONF_THRESHOLD`: confidence threshold for detections
- `IMGSZ`: input image size for inference
- `FLIP_FRAME`: set `True` if the camera image is mirrored

## Notes

- Press `q` to exit the application.
- If the camera does not open, try changing `CAMERA_INDEX` from `1` to `0`, `2`, or `3`.
- On Windows, `cv2.CAP_DSHOW` may help if camera initialization is unstable.

## Portfolio / CV Positioning

If you want this repository to look stronger on your CV, keep the README focused on measurable value. You should add:

- the problem you solved
- the model or dataset source
- the hardware used for testing
- example classes detected
- average FPS or runtime observations
- 1 short paragraph describing your contribution if this was part of a larger project

Suggested resume summary:

> Built a real-time dashcam object detection demo using Python, OpenCV, and Ultralytics YOLO; integrated live camera inference, bounding-box visualization, and FPS monitoring for practical computer vision deployment.

## Future Improvements

- support video file input in addition to webcam input
- save annotated output video
- export model to ONNX for lighter deployment
- add CLI arguments for camera index and confidence threshold
- report per-frame latency and detection counts

## Author

**Le Thanh Hai**

- GitHub: [ThanhHaipear](https://github.com/ThanhHaipear)
