from flask import Flask, render_template, Response, request
import numpy as np
import cv2 as cv
import time

f = cv.FileStorage('calibrate.xml', cv.FILE_STORAGE_READ)
intrinsic = f.getNode('intrinsic').mat()
distortion = f.getNode('distortion').mat()
app = Flask(__name__)
t_errors = []
r_errors = []
video_path = ""

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_center_pos(corners):
    corners = np.array(corners)
    center_pixels = []

    for i in range(len(corners)):
        x_center = np.sum(corners[i, 0, :, 0]) / 4
        y_center = np.sum(corners[i, 0, :, 1]) / 4
        center_pixels.append((x_center, y_center))

    return center_pixels

def pose_matching(demo, user):
    t_error = 0
    r_error = 0

    # no markers detected in demo video
    if demo[0][0] is None:
        t_errors.append(50)
        r_errors.append(3)
        return

    # no markers detected in user video
    if user[0][0] is None:
        t_errors.append(50)
        r_errors.append(3)
        return

    # demo[0] = ids, demo[1] = tvec[], demo[2] = rvec[]
    demo = np.array(demo, dtype=object)
    demo = demo[:, demo[0, :].argsort()]
    user = np.array(user, dtype=object)
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
            # cv.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # cv.aruco.drawAxis(frame, intrinsic, distortion, rvec, tvec, 5)

            # Put text
            # frame = cv.putText(frame, 'id: ' + str(ids[i]) + ', z: ' + str(int(tvec[2])), tuple(corners[i][0][0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
    center_poses = np.array(get_center_pos(corners))
    return frame, ids, t, r, center_poses

def draw(frame, n, start, end):
    for pid in range(start, end + 1):
        pixel = np.load(f'path/path{pid}.npy')
        for i in range(len(pixel)):
            if i >= n:
                break
            p = pixel[i]
            frame = cv.circle(frame, (p[0], p[1]), 5, (0, 255 * i / n, 0), -1)

    return frame

def draw_marker_poses(frame, demo_ids, demo_marker_poses, user_ids, user_marker_poses):
    groups = [[3, 4, 11, 13],   # left hand
            [2, 6, 1, 0],     # right hand 
            [8, 12, 14, 15],  # left foot 
            [5, 7, 9, 10]]    # right foot 

    if video_path == 'demos/bird_demo.mp4':
        frame = draw(frame, len(demo_ids), 1, 4)
    elif video_path == 'demos/leg_demo.mp4':
        frame = draw(frame, len(demo_ids), 5, 8)
    else:
        # draw demo ground trut
        path = [[], [], [], []]
        n = len(demo_marker_poses)
        for fid in range(len(demo_marker_poses)):
            ids = demo_ids[fid]
            for j in range(len(groups)):
                id_group = groups[j]
                pos = [0, 0]
                cnt = 0
                for i in range(len(ids)):
                    if ids[i] in id_group:
                        cnt += 1
                        pos += demo_marker_poses[fid][i]
                if cnt == 0:
                    continue
                path[j].append((int(pos[0] / cnt), int(pos[1] / cnt)));

        for group_id in range(len(groups)):
            p = path[group_id]
            if len(p) == 1:
                continue
            n = len(p)
            for i in range(len(p) - 1):
                p1 = p[i]
                p2 = p[i + 1]
                frame = cv.line(frame, p1, p2, (0, 255 * i / n, 0), thickness=5)
            

    # draw user ground truth
    path = [[], [], [], []]
    n = len(user_marker_poses)
    for fid in range(len(user_marker_poses)):
        ids = user_ids[fid]
        for j in range(len(groups)):
            id_group = groups[j]
            pos = [0, 0]
            cnt = 0
            for i in range(len(ids)):
                if ids[i] in id_group:
                    cnt += 1
                    pos += user_marker_poses[fid][i]
            if cnt == 0:
                continue
            path[j].append((int(pos[0] / cnt), int(pos[1] / cnt)));

    for group_id in range(len(groups)):
        p = path[group_id]
        if len(p) == 1:
            continue
        n = len(p)
        for i in range(len(p) - 1):
            p1 = p[i]
            p2 = p[i + 1]
            frame = cv.line(frame, p1, p2, (0, 0, 255 * i / n), thickness=5)

    return frame

def evaluation():
    alpha = 0.1
    error = alpha * np.sum(t_errors) + (1 - alpha) * np.sum(r_errors)
    print(f'error: {error}')

    threshold = 100

    if error > threshold:
        print('Your action is wrong!!')
    else:
        print('Your action is correct!!')

def gen_frames():
    global video_path
    video_path = ""
    with open('option.txt') as f:
        lines = f.readlines()
        video_path = lines[0]

    print(video_path)
    demo = cv.VideoCapture(video_path)
    user = cv.VideoCapture(0)
    time.sleep(2.0)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(f'./users/user.avi', fourcc, 20.0, (1280,  720))

    while True:
        ret, frame_user = user.read()
        
        # can't read from camera
        if not ret:
            break

        frame_user = cv.flip(frame_user, 1)

        ret, frame_demo = demo.read()  # frame (720, 1280, 3)

        # demo video ended
        if not ret:
            # determine action correct or not. If not, save replay frames
            out.release()
            time.sleep(1)

            evaluation()

            # do replay ----------------------------------------------------
            while True:
                replay = cv.VideoCapture('./users/user.avi')
                demo = cv.VideoCapture(video_path)

                demo_marker_poses = [] # (frame_num, marker_num, (x, y))
                user_marker_poses = []
                demo_ids = [] # (frame_num, marker_num, id)
                user_ids = []
                while True:
                    ret, frame_user = replay.read()  # frame (720, 1280, 3)
                    # replay video ended
                    if not ret:
                        print('replay user failed')
                        break

                    ret, frame_demo = demo.read()  # frame (720, 1280, 3)

                    # replay video ended
                    if not ret:
                        print('replay demo failed')
                        break

                    ret, live_user = user.read()

                    # replay video ended
                    if not ret:
                        print('replay live failed')
                        break

                    _, ids, _, _, poses = pose_esitmation(frame_demo)
                    demo_marker_poses.append(poses)
                    demo_ids.append(ids)
                    _, ids, _, _, poses = pose_esitmation(frame_user)
                    user_marker_poses.append(poses)
                    user_ids.append(ids)

                    output = draw_marker_poses(frame_user, demo_ids, demo_marker_poses, user_ids, user_marker_poses)

                    # show replay video
                    cv.imshow('replay', output)

                    blank = np.zeros((720, 90, 3), np.uint8)
                    blank.fill(255)
                    output = cv.hconcat([live_user, blank, output])
                    _, outbuf = cv.imencode('.jpg', output)
                    outbuf = outbuf.tobytes()
                    yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + outbuf + b'\r\n')

                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                replay.release()
                cv.destroyWindow('replay')
            # do replay ----------------------------------------------------

            # reset
            out = cv.VideoWriter(f'./users/user.avi', fourcc, 20.0, (1280,  720))
            t_errors.clear()
            r_errors.clear()
            demo = cv.VideoCapture(video_path)
            continue

        output_demo, ids, tvec, rvec, _ = pose_esitmation(frame_demo)
        demo_data = [ids, tvec, rvec]
        output_user, ids, tvec, rvec, _ = pose_esitmation(frame_user)
        user_data = [ids, tvec, rvec]
        pose_matching(demo_data, user_data)

        cv.imshow('demo', output_demo)
        cv.imshow('user', output_user)

        # save user video for replay
        out.write(output_user)

        blank = np.zeros((720, 90, 3), np.uint8)
        blank.fill(255)
        output = cv.hconcat([output_user, blank, output_demo])
        _, outbuf = cv.imencode('.jpg', output)
        outbuf = outbuf.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + outbuf + b'\r\n')

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    demo.release()
    user.release()
    out.release()
    cv.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/',methods=['POST','GET'])
def index():
    global init_flg
    if request.method == 'POST' and init_flg == False:
        if request.form.get('action'):
            demo_path = request.form.get('action')
            demo_path = f'demos/{demo_path}_demo.mp4'
            print('demo_path', demo_path)
            with open('option.txt', 'w') as f:
                f.write(demo_path)

            return render_template('index.html', status="demo")

    else:
        init_flg = False
    return render_template('index.html', status="init")

if __name__ == '__main__':
    global init
    init_flg = True
    app.run('0.0.0.0')
