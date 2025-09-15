import cv2
import numpy as np

def detect_red_and_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = mask_red | mask_yellow

    red_objects = cv2.bitwise_and(frame, frame, mask=mask_red)
    yellow_objects = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Red Object', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_yellow:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, 'Yellow Object', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)

    return frame, red_objects, yellow_objects, combined_mask

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detected_frame, red_only, yellow_only, mask = detect_red_and_yellow(frame)
    cv2.imshow('Deteksi Merah & Kuning', detected_frame)
    cv2.imshow('Mask Gabungan', mask)
    cv2.imshow('Hanya Merah', red_only)
    cv2.imshow('Hanya Kuning', yellow_only)

cap.release()
cv2.destroyAllWindows()
