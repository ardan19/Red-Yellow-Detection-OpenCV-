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
```
How to run:
- Download or clone the file
- Create and open directory named "Color Detection" and place the downloaded or cloned file named "script.py" to this directory
- Open console command or terminal in this path
- install dependencies using console command or terminal
- type "py script.py" to use the script in the console command or terminal
- Pop up will show and start to detecting color red and yellow
- press "q" to exit the window detection

## How the code works
### Import Libraries
```bash
import cv2
import numpy as np
```
cv2 = OpenCV for computer vision tasks.
numpy = for handling arrays (used in HSV range definitions).
### Define Detection Function
```bash
def detect_red_and_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```
Converts each frame from BGR → HSV.
HSV is chosen because it separates color from brightness.

### Define HSV Ranges
```bash
# Red ranges
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Yellow range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
```
Red requires two ranges due to wrapping in HSV space.
Yellow is within a single range.
### Create Masks
```bash
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 | mask_red2
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
```
cv2.inRange creates a binary image (mask):
- White = pixels inside range.
- Black = everything else.
### Apply Mask to Frame
```bash
red_objects = cv2.bitwise_and(frame, frame, mask=mask_red)
yellow_objects = cv2.bitwise_and(frame, frame, mask=mask_yellow)
```
Keeps only red or yellow areas, everything else turns black.
### Find and Draw Contours
```bash
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_red:
    if cv2.contourArea(cnt) > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Red Object', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
```
Finds object boundaries in the mask.
Filters out small noise (area > 500).
Draws bounding boxes + labels.
Same logic applies for yellow objects.
### Video Capture Loop
```bash
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    detected_frame, red_only, yellow_only, mask = detect_red_and_yellow(frame)

    cv2.imshow('Red & Yellow Detection', detected_frame)
    cv2.imshow('Combined Mask', mask)
    cv2.imshow('Red Only', red_only)
    cv2.imshow('Yellow Only', yellow_only)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
Captures frames from webcam.
Calls detection function.
Displays four windows with results.
Press q to quit.
### Cleanup
```bash
cap.release()
cv2.destroyAllWindows()
```
Releases webcam and closes all OpenCV windows.
