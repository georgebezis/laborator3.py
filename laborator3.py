import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

left_top=[0,0]
left_bottom=[0,0]
right_top=[0,0]
right_bottom=[0,0]
while True:
    ret, frameclr = cam.read()
    frame_h=400
    frame_w=640
    frameclr = cv2.resize(frameclr,(frame_w,frame_h))
    frame = cv2.cvtColor(frameclr, cv2.COLOR_BGR2GRAY)
    trapez=np.zeros(frame.shape,dtype=np.uint8)

    upper_left=(int(frame_w*0.457),int(frame_h*0.769))
    upper_right=(int(frame_w*0.51),int(frame_h*0.769))
    lower_left=(0,frame_h-int(frame_h*0.0))
    lower_right=(frame_w,frame_h-int(frame_h*0.0))
    bounds_trapez=np.array([upper_right,upper_left,lower_left,lower_right],dtype=np.int32)
    trapez=cv2.fillConvexPoly(trapez,bounds_trapez,1)
    road=trapez*frame
    bounds_trapez=np.float32(bounds_trapez)
    bounds_screen=np.array([(frame_w,0),(0,0),(0,frame_h),(frame_w,frame_h)])
    bounds_screen=np.float32(bounds_screen)
    magic_matrix=cv2.getPerspectiveTransform(bounds_trapez,bounds_screen)
    stretched=cv2.warpPerspective(road,magic_matrix,(frame_w,frame_h))
    blur=cv2.blur(stretched,(11,11))
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
    thresholdframe_copy[0:frame_h,0:int(frame_w*0.032)]=0
    thresholdframe_copy[0:frame_h,frame_w-int(frame_w * 0.032):frame_w] = 0
    thresholdframe_copy[frame_h-int(frame_h*0.03):frame_h, 0:frame_w] = 0
    thresholdframe_copy[0:int(frame_h * 0.05), 0:frame_w] = 0
    #thresholdframe_copy[int(0.001*400):400,0:640]=0

    leftpart=np.copy(thresholdframe_copy[0:frame_h,0:int(frame_w/2)])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    leftpart= cv2.dilate(leftpart, kernel, iterations=3)
    leftpart = cv2.convertScaleAbs(leftpart)
    cv2.imshow("leftpart", leftpart)
    leftpartpoints=np.argwhere(leftpart)
    rightpart=np.copy(thresholdframe_copy[0:frame_h,int(frame_w/2):frame_w])
    rightpartpoints=np.argwhere(rightpart)

    #print(rightpart[0])


    left_xs=leftpartpoints[:,1]
    left_ys=leftpartpoints[:,0]
    right_xs=rightpartpoints[:,1]+frame_w/2
    right_ys=rightpartpoints[:,0]
    #10
    lb,la=np.polynomial.polynomial.polyfit(left_xs,left_ys,1)
    rb,ra=np.polynomial.polynomial.polyfit(right_xs,right_ys,1)


    left_top_y=0 #left_ys[0]
    left_top_x=(left_top_y-lb)/la
    left_bottom_y=frame_h #left_ys[int(len(left_ys)-1)]
    left_bottom_x=(left_bottom_y-lb)/la

    right_top_y =0
    right_top_x = (right_top_y - rb) / ra
    right_bottom_y = frame_h
    right_bottom_x = (right_bottom_y- rb) / ra


         #c

    if (-(10**8)) < left_top_x < (10 ** 8):
     left_top=[int(left_top_x),int(left_top_y)]

    if (-(10**8)) < left_bottom_x < 10**8:
     left_bottom = [int(left_bottom_x), int(left_bottom_y)]
    if (-(10**8)) < right_top_x < (10**8):
     right_top = [int(right_top_x), int(right_top_y)]
    if (-(10**8)) < right_bottom_x < (10**8):
     right_bottom = [int(right_bottom_x), int(right_bottom_y)]

    print(right_top)
    #print(right_bottom)
    #print(left_bottom)
    #print(left_top)
    lines=np.copy(thresholdframe_copy)
    lines=cv2.line(lines,left_top,left_bottom,(200,0,0),10)
    lines=cv2.line(lines,right_top,right_bottom,(100,0,0),5)
    lines=cv2.line(lines,(int(640/2),0),(int(640/2),400),(255,0,0),1)


    roadlinesleft=np.zeros(frame.shape)
    roadlinesleft=cv2.line(roadlinesleft, left_top, left_bottom, (250, 0, 0), 3)

    magic_matrix2=cv2.getPerspectiveTransform(bounds_screen,bounds_trapez)
    roadlinesleft=cv2.warpPerspective(roadlinesleft,magic_matrix2,(frame_w,frame_h))
    leftcoord=np.argwhere(roadlinesleft)

    roadlinesright = np.zeros(frame.shape)
    roadlinesright = cv2.line(roadlinesright, right_top, right_bottom, (250, 0, 0), 3)

    magic_matrix2 = cv2.getPerspectiveTransform(bounds_screen, bounds_trapez)
    roadlinesright = cv2.warpPerspective(roadlinesright, magic_matrix2, (frame_w, frame_h))
    rightcoord = np.argwhere(roadlinesright)

    frame_copy = np.copy(frameclr)
    collor_left=(50,50,250)
    collor_right=(50,250,50)

    frame_copy[leftcoord[:,0],leftcoord[:,1]]=collor_left
    frame_copy[rightcoord[:, 0], rightcoord[:, 1]] = collor_right
   # frame_copy=cv2.applyColorMap(frame_copy,leftcoord,collor_left)
    #print(leftcoord)
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
    cv2.imshow("roadlines",frame_copy)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

