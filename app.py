from ultralytics import YOLO
import numpy as np
import cv2 as cv

model = YOLO('yolov8n.pt')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display the resulting frame
    if cv.waitKey(1) == ord('q'):
        break
    model.predict(frame, show=True, conf=0.5)
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
