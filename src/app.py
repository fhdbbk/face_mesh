import cv2
import dlib
import numpy as np
from pathlib import Path
import os

BASE_DIR = Path(os.path.abspath(__file__)).parent.parent

p = BASE_DIR / "models/dlib/shape_predictor_68_face_landmarks.dat"

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	
	return coords

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(p))
cap = cv2.VideoCapture(0)
while True:
  
    _, image = cap.read()
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()