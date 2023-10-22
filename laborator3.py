import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(640,400))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame2=np.zeros(frame.shape,dtype=np.uint8)

    upper_left=(int(640*0.45),int(400*0.73))
    upper_right=(int(640*0.55),int(400*0.73))
    lower_left=(0,400)
    lower_right=(640,400)
    points_trapez=np.array([upper_right,upper_left,lower_left,lower_right],dtype=np.int32)
    frame2=cv2.fillConvexPoly(frame2,points_trapez,1)
    frame3=frame2*frame
    print("laptop")

    if ret is False:

        break
    cv2.imshow('Original', frame)
   # cv2.imshow('trapez',frame2)
    cv2.imshow('road',frame3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

