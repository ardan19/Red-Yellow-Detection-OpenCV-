# Red & Yellow Color Detection with OpenCV

This project demonstrates **real-time color detection** using OpenCV in Python.  
It detects **red** and **yellow objects** from a webcam feed, draws bounding boxes, and labels them on the screen.

---

## Features
- Detects **red** and **yellow** colors in real-time.  
- Uses **HSV color space** for more accurate color segmentation.  
- Draws **bounding boxes** and labels on detected objects.  
- Shows multiple outputs:
  - Original frame with detection.
  - Binary mask of detected areas.
  - Isolated red regions.
  - Isolated yellow regions.

---

##  Requirements
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy  

Install dependencies:
```bash
pip install opencv-python numpy

How to run:
- Download or clone the file
- Create and open directory named "Color Detection"
- Open console command or terminal in this path
- install dependencies using console command or terminal
- type "py script.py" to use the script in the console command or terminal
- Pop up will show and start to detecting color red and yellow
- press "q" to exit the window detection

## How the code works
```bash
import cv2
import numpy as np

cv2 = OpenCV for computer vision tasks.
numpy = for handling arrays (used in HSV range definitions).

```bash
def detect_red_and_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

Converts each frame from BGR → HSV.
HSV is chosen because it separates color from brightness.
