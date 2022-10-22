from distutils.log import error
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import numpy as np
import cv2 as cv
import time
import math

f = cv.FileStorage('calibrate.xml', cv.FILE_STORAGE_READ)
intrinsic = f.getNode('intrinsic').mat()
distortion = f.getNode('distortion').mat()
app = Flask(__name__)
t_errors = []
r_errors = []

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def pose_matching(demo, user):
    t_error = 0
    r_error = 0

    # no markers detected in demo video
    if demo[0][0] is None:
        return

    # no markers detected in user video
    if user[0][0] is None:
        t_errors.append(50)
        r_errors.append(3)
        return

    # demo[0] = ids, demo[1] = tvec[], demo[2] = rvec[]
    demo = np.array(demo)
    demo = demo[:, demo[0, :].argsort()]
    user = np.array(user)
    user = user[:, user[0, :].argsort()]
    
    j = 0
    for i in range(len(demo[0])):
        while j < len(user[0])  and demo[0][i] < user[0][j]:
            j += 1
        
        # can't find demo marker id[i] in user marker ids
        if j >= len(user[0]) or demo[0][i] != user[0][j]:
            break

        target_tvec = demo[1][i]
        target_rvec = demo[2][i]
        user_tvec = user[1][j]
        user_rvec = user[2][j]

        # translation error
        d = user_tvec - target_tvec
        t_error += np.sqrt(np.sum(np.square(d)))
        
        # rotation error
        theta = angle_between(user_rvec, target_rvec)
        r_error += theta
        e = np.linalg.norm(user_rvec) - np.linalg.norm(target_rvec)
        r_error += e

    print(f'translation error: {t_error}, rotation error: {r_error}')
    t_errors.append(t_error)
    r_errors.append(r_error)

    return

def pose_esitmation(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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

        # demo video ended
        if not ret:
            demo = cv.VideoCapture('./demos/demo.avi')
            # evaluation()
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

def demo_gen_frames(frames):
    counter = 0
    length = len(frames)
    while counter < length:
        frame = frames[counter]
        counter += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.5)
    global result
    result = False

@app.route('/is_decoded/')
def is_decoded():
    global result
    return jsonify({'is_decoded': result})


@app.route('/video_feed')
def video_feed():
    global frames 
    return Response(demo_gen_frames(frames), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/demo_video_feed')
def demo_video_feed():
    global frames
    return Response(demo_gen_frames(frames), mimetype='multipart/x-mixed-replace; boundary=frame')

frames_repo = {'1':['1', '2', '3', '4', '5'], 
                '2':['6', '7', '8', '9', '0'],
                '3':['3', '4', '5', '6', '7']}
                    
@app.route('/',methods=['POST','GET'])
def index():
    global frames
    global result
    res = ''
    if request.form:
        result = False
        res = '1'
        frames = [open('./static/images/' + f + '.jpg', 'rb').read() for f in frames_repo[request.form['action']]]
    return render_template('index.html', result=True)

# @app.route('/result',methods=['POST','GET'])
# def result():
#     return render_template('index.html', result="")

if __name__ == '__main__':
    app.run('0.0.0.0')
