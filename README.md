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
