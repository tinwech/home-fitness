from flask import Flask, render_template, Response
import numpy as np
import cv2 as cv
import time
import math

f = cv.FileStorage('calibrate.xml', cv.FILE_STORAGE_READ)
intrinsic = f.getNode('intrinsic').mat()
distortion = f.getNode('distortion').mat()
app = Flask(__name__)


def pose_esitmation(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
    cv.aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv.aruco.detectMarkers(
        gray, cv.aruco_dict, parameters=parameters)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(
                corners[i], 7.4, intrinsic, distortion)
            rvec = rvec.flatten('F')
            tvec = tvec.flatten('F')

            # Draw a square around the markers
            cv.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 5)

            # Put text
            frame = cv.putText(frame, 'z: ' + str(tvec[2]), tuple(
                corners[i][0][0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame


frames = [open('./static/images/' + f + '.jpg', 'rb').read()
          for f in ['1', '2', '3', '4', '5']]


def gen_frames():
    counter = 0
    while True:
        frame = frames[counter % 5]
        counter += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.5)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')
    # return "hello world"


if __name__ == '__main__':

    video = cv.VideoCapture(1)
    time.sleep(2.0)
    app.run('0.0.0.0')
   while True:
        ret, frame = video.read()  # frame (480, 640, 3)

        if not ret:
            break

        output = pose_esitmation(frame)

        cv.imshow('output', output)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
