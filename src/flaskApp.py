from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)

# To overcome the problem of "can't open camera by index" when using debug mode
# https://stackoverflow.com/questions/61047207/opencv-videocapture-does-not-work-in-flask-project-but-works-in-basic-example
if os.environ.get('WERKZEUG_RUN_MAIN') or Flask.debug is False:
    cap = cv2.VideoCapture(0)


def gen_frames(camera):  
    # print("reaching here")
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
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