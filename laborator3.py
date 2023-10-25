import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(640,400))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    trapez=np.zeros(frame.shape,dtype=np.uint8)

    upper_left=(int(640*0.45),int(400*0.79))
    upper_right=(int(640*0.55),int(400*0.79))
    lower_left=(0,400)
    lower_right=(640,400)
    bounds_trapez=np.array([upper_right,upper_left,lower_left,lower_right],dtype=np.int32)
    cv2.fillConvexPoly(trapez,bounds_trapez,1)
    road=trapez*frame
    bounds_trapez=np.float32(bounds_trapez)
    bounds_screen=np.array([(640,0),(0,0),(0,400),(640,400)])
    bounds_screen=np.float32(bounds_screen)
    magic_matrix=cv2.getPerspectiveTransform(bounds_trapez,bounds_screen)
    stretched=cv2.warpPerspective(road,magic_matrix,(640,400))
    blur=cv2.blur(stretched,(3,3))
    sobel_vertical=np.float32([[-1, -2, -1],[0,0,0],[1,2,1]])
    sobel_horizontal=np.transpose(sobel_vertical)
    blur=np.float32(blur)
    sobelHorizontal=cv2.filter2D(blur,-1,sobel_horizontal,)
    sobelVertical=cv2.filter2D(blur,-1,sobel_vertical)
    sobel=((sobelVertical)**2+(sobelHorizontal)**2)**1/2
    threshold=int(255/2)
    thresholdframe=np.where(sobel<threshold,0,255)
    thresholdframe_copy=np.copy(thresholdframe)
    thresholdframe_copy[0:400,0:int(640*0.05)]=0
    thresholdframe_copy[0:400,640-int(640 * 0.05):640] = 0
    leftpart=np.argwhere(thresholdframe_copy[0:400,0:int(640/2)]>1)
    rightpart=np.argwhere(thresholdframe_copy[0:400,int(640/2):640])

    print(rightpart[0])


    #left_xs=
    #left_ys=
    #right_xs=
    #right_ys=



    sobelHorizontal=cv2.convertScaleAbs(sobelHorizontal)
    sobelVertical = cv2.convertScaleAbs(sobelVertical)
    blur=cv2.convertScaleAbs(blur)
    sobel=cv2.convertScaleAbs(sobel)
    thresholdframe=cv2.convertScaleAbs(thresholdframe)
    thresholdframe_copy = cv2.convertScaleAbs(thresholdframe_copy)
    if ret is False:

        break
    cv2.imshow('Original', frame)
    cv2.imshow('trapez',trapez)
    cv2.imshow('road',road )
    cv2.imshow("stretched",stretched)
    cv2.imshow('blured',blur)
    cv2.imshow("sobelhorizontal",sobelHorizontal)
    cv2.imshow("sobelvertical", sobelVertical)
    cv2.imshow("sobel", sobel)
    cv2.imshow("thresh",thresholdframe )
    cv2.imshow("thresh_copy", thresholdframe_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

