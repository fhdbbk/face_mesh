from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
import os
from pathlib import Path

BASE_DIR = Path(os.path.abspath(__file__)).parent.parent

p = BASE_DIR / "models/dlib/shape_predictor_68_face_landmarks.dat"

app = Flask(__name__)

# To overcome the problem of "can't open camera by index" when using debug mode
# https://stackoverflow.com/questions/61047207/opencv-videocapture-does-not-work-in-flask-project-but-works-in-basic-example
if os.environ.get('WERKZEUG_RUN_MAIN') or Flask.debug is False:
    cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(p))

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	
	return coords


def gen_frames(camera):  
    # print("reaching here")
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for (i, rect) in enumerate(rects):
                x1 = rect.left()
                y1 = rect.top()
                x2 = rect.right()
                y2 = rect.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                shape = predictor(gray, rect)
                shape = shape_to_np(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)