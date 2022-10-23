from unittest.mock import patch
import cv2
import numpy as np 

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

path = []

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(frame,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)
            pt1_x,pt1_y=x,y
            path.append([x, y])
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(frame,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)        


cv2.namedWindow('output')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('demos/leg_demo.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# def draw(frame):
#     pixel = np.load('path1.npy')
#     for p in pixel:
#         frame = cv2.circle(frame, (p[0], p[1]), 5, (0, 255, 0), -1)
#     pixel = np.load('path2.npy')
#     for p in pixel:
#         frame = cv2.circle(frame, (p[0], p[1]), 5, (0, 255, 0), -1)
#     pixel = np.load('path3.npy')
#     for p in pixel:
#         frame = cv2.circle(frame, (p[0], p[1]), 5, (0, 255, 0), -1)
#     return frame
    
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    cv2.setMouseCallback('output',line_drawing)

    # frame = draw(frame)
    # Display the resulting frame
    cv2.imshow('output',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
path = np.array(path)
np.save('path4.npy', path)

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()