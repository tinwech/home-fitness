import cv2

cap = cv2.VideoCapture(1)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
i = input()
demo = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 水平上下翻轉影像
    #frame = cv2.flip(frame, 0)
    # write the flipped frame
    if demo:
        out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == ord('d'):
        demo = True 
        out = cv2.VideoWriter(f'./demos/demo_{i}.avi', fourcc, 20.0, (640,  480))
    elif cv2.waitKey(30) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()