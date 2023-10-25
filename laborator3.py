import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
aux_left_top = 0
aux_left_bottom = 0
aux_right_top = 0
aux_right_bottom = 0
left_top=[0,0]
left_bottom=[0,0]
right_top=[0,0]
right_bottom=[0,0]
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(640,400))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    trapez=np.zeros(frame.shape,dtype=np.uint8)

    upper_left=(int(640*0.45),int(400*0.78))
    upper_right=(int(640*0.55),int(400*0.78))
    lower_left=(0,380)
    lower_right=(640,380)
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
    sobel=np.sqrt((sobelVertical)**2+(sobelHorizontal)**2)
    threshold=int(255/10)
    #thresholdframe=np.where(sobel<threshold,0,255)
    aux,thresholdframe=cv2.threshold(sobel,threshold,255,cv2.THRESH_BINARY)
    #9
    thresholdframe_copy=np.copy(thresholdframe)
    thresholdframe_copy[0:400,0:int(640*0.05)]=0
    thresholdframe_copy[0:400,640-int(640 * 0.05):640] = 0
    #thresholdframe_copy[int(0.001*400):400,0:640]=0
    leftpart=np.argwhere(thresholdframe_copy[0:400,0:int(640/2)]>1)
    rightpart=np.argwhere(thresholdframe_copy[0:400,int(640/2):640])

    #print(rightpart[0])


    left_xs=leftpart[:,1]
    left_ys=leftpart[:,0]
    right_xs=rightpart[:,1]+640/2
    right_ys=rightpart[:,0]
    #10
    lb,la=np.polynomial.polynomial.polyfit(left_xs,left_ys,1)
    rb,ra=np.polynomial.polynomial.polyfit(right_xs,right_ys,1)


    left_top_y=0
    left_top_x=(left_top_y-lb)/la
    left_bottom_y=400
    left_bottom_x=(left_bottom_y-lb)/la

    right_top_y = 0
    right_top_x = (right_top_y - rb) / ra
    right_bottom_y = 400
    right_bottom_x = (right_bottom_y- rb) / ra


         #c

    aux_left_top = left_top_x

    aux_left_bottom = left_bottom_x

    aux_right_top = right_top_x

    aux_right_bottom = right_bottom_x
    if (-(10 **3)) < left_top_x < (10 ** 3):
     left_top=[int(aux_left_top),int(left_top_y)]
    if (-(10 ** 3)) < left_bottom_x < (10 ** 3):
     left_bottom = [int(aux_left_bottom), int(left_bottom_y)]
    if (-(10 ** 3)) < right_top_x < (10 ** 3):
     right_top = [int(aux_right_top), int(right_top_y)]
    if (-(10 ** 3)) < right_bottom_x < (10 ** 3):
     right_bottom = [int(aux_right_bottom), int(right_bottom_y)]

    print(right_top)

    lines=np.copy(thresholdframe_copy)
    lines=cv2.line(lines,left_top,left_bottom,(200,0,0),10)
    lines=cv2.line(lines,right_top,right_bottom,(100,0,0),5)
    #lines=cv2.line(lines,(400,int(640*0.5)),(0,int(640*0.5)),(255,0,0),1)

    sobelHorizontal=cv2.convertScaleAbs(sobelHorizontal)
    sobelVertical = cv2.convertScaleAbs(sobelVertical)
    blur=cv2.convertScaleAbs(blur)
    sobel=cv2.convertScaleAbs(sobel)
    thresholdframe=cv2.convertScaleAbs(thresholdframe)
    thresholdframe_copy = cv2.convertScaleAbs(thresholdframe_copy)
    lines=cv2.convertScaleAbs(lines)
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
    cv2.imshow("lines", lines)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

