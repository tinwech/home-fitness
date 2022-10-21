from flask import Flask, render_template, Response
import numpy as np
import cv2 as cv
import time
import math

f = cv.FileStorage('calibrate.xml', cv.FILE_STORAGE_READ)
intrinsic = f.getNode('intrinsic').mat()
distortion = f.getNode('distortion').mat()
app = Flask(__name__)


def pose_matching(demo, user):
    if demo[0][0] is None:
        return

    demo = np.array(demo)
    demo = demo[:, demo[0, :].argsort()]
    user = np.array(user)
    user = user[:, user[0, :].argsort()]
    
    # print(user)
    # for i in range(len(ids)):
    #     print(ids[i]) 
    #     print(tvec[i])
    #     print(rvec[i])
    return

def pose_esitmation(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
    cv.aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict, parameters=parameters)
    ids = np.array(ids).flatten('F')

    t = []
    r = []
    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 7.4, intrinsic, distortion)
            rvec = rvec.flatten('F')
            tvec = tvec.flatten('F')
            r.append(rvec)
            t.append(tvec)

            # Draw a square around the markers
            cv.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 5)

            # Put text
            frame = cv.putText(frame, 'id: ' + str(ids[i]) + 'z: ' + str(int(tvec[2])), tuple(corners[i][0][0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
    return frame, ids, t, r


def gen_frames():
    demo = cv.VideoCapture('./demos/demo_1.avi')

    user = cv.VideoCapture(1)
    time.sleep(2.0)

    # while demo.isOpened():
    while True:
        ret, frame_demo = demo.read()  # frame (480, 640, 3)

        if not ret:
            demo = cv.VideoCapture('./demos/demo_1.avi')
            continue

        ret, frame_user = user.read()

        if not ret:
            break

        output_demo, ids, tvec, rvec = pose_esitmation(frame_demo)
        demo_data = [ids, tvec, rvec]
        output_user, ids, tvec, rvec = pose_esitmation(frame_user)
        user_data = [ids, tvec, rvec]

        pose_matching(demo_data, user_data)

        cv.imshow('output demo', output_demo)
        cv.imshow('output user', output_user)

        _, img = cv.imencode('.jpg', output_user)
        img = img.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    demo.release()
    cv.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0')
