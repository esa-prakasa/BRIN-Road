"""
Created on Mon Mar  8 07:45:56 2021
@author: Elli A Gojali
"""
import cv2
import math
import numpy as np
import time

data1 = []
data2 = []
data3 = []
data1l = []
data2l = []
data3l = []
dataTotal = []

offset = 375

cap = cv2.VideoCapture('NO20201130-110302-000138.MP4')
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('LaneDetect011122.mp4',fourcc,20.0,(1280,720))
 
if (cap.isOpened()== False):
  print("Error opening video stream or file")
while(cap.isOpened()):
  ret, frame = cap.read()

  if ret == True:
 
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    crop_img = img[375:720, 0:1280]
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(crop_img)[0] #Position 0 of the returned tuple are the detected lines
    xa = (img.shape[1])
    ya = (img.shape[0])

    dmax = math.sqrt(((xa*xa)/4)+(ya*ya))      
    dmaxl = dmax

    ax1 = xa
    ay1 = 0
    ax2 = xa
    ay2 = 0
    gdt = 0
    gdtl = 0
            
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if (y1<y2):
            a=x1
            b=y1
            x1=x2
            y1=y2
            x2=a
            y2=b
            
            dkuad = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)
            if (dkuad > 10000):
                if (x1>(xa/2)): #right side
                    gd = (y1-y2)/(x1-x2)
                    if (0.4<gd<2):
                        data1.append(line)
                        dx = (x1+x2)/2
                        dy = (y1+y2)/2
                        dt = math.sqrt((dx-(xa/2))*(dx-(xa/2)) + (dy-ya)*(dy-ya)) #jarak titik tengan bawah dan garis
                        if (dt<dmax):
                            gdt = gd
                            dmax=dt
                            ax1 = int(x1)
                            ay1 = int(y1)#+offset
                            ax2 = int(x2)
                            ay2 = int(y2)#+offset
    
                if (x1<(xa/2)): #left side
                    gdl = (y1-y2)/(x1-x2)
                    if (-2<gdl<-0.4):
                        data1l.append(line)
                        dxl = (x1+x2)/2
                        dyl = (y1+y2)/2
                        dtl = math.sqrt((dxl-(xa/2))*(dxl-(xa/2)) + (dyl-ya)*(dyl-ya))
                        if (dtl<dmaxl):
                            gdtl = gdl
                            dmaxl=dtl
                            ax1l = int(x1)
                            ay1l = int(y1)#+offset
                            ax2l = int(x2)
                            ay2l = int(y2)#+offset                   
    data2 = np.array(data1)
    data3 = np.array([[ax1,ay1,ax2,ay2]])
    data2l = np.array(data1l)
    data3l = np.array([[ax1l,ay1l,ax2l,ay2l]])
    dataAkh = data2 + data3
    dataTot1 = ([ax1l,ay1l,ax2l,ay2l],[ax1,ay1,ax2,ay2])
    dataTotal1 = np.array(dataTot1)
    parametersR = np.polyfit((ax1, ax2), (ay1, ay2), 1)
    parametersL = np.polyfit((ax1l, ax2l), (ay1l, ay2l), 1)
    
    if (parametersR[0] != 0):
        ty1 = crop_img.shape[0]
        ty2 = int(ty1*(3/10))
        tx1R = int((ty1 - parametersR[1])/parametersR[0]) #intercept)/slope)
        tx2R = int((ty2 - parametersR[1])/parametersR[0]) #intercept)/slope)
        tx1L = int((ty1 - parametersL[1])/parametersL[0]) #intercept)/slope)
        tx2L = int((ty2 - parametersL[1])/parametersL[0]) #intercept)/slope)

    dataTot = ([tx1L,ty1+offset,tx2L,ty2+offset], [tx1R,ty1+offset,tx2R,ty2+offset])
    dataTotal = np.array(dataTot)
    drawn_imgA = lsd.drawSegments(crop_img,lines) #dataTotal
    drawn_filter = lsd.drawSegments(crop_img,dataTotal1) #dataTotal    

    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.line(blank_image, (tx1L,ty1+offset), (tx2L,ty2+offset), (0, 255, 0), thickness=10)
    cv2.line(blank_image, (tx1R,ty1+offset), (tx2R,ty2+offset), (0, 255, 0), thickness=10)
    imgR = cv2.addWeighted(frame, 1, blank_image, 0.6, 0.0) #blank_image
    cv2.imshow('Lane Detection', imgR)    
    cv2.imshow('Lane Source', img)
    cv2.imshow('Image RoI & LSD', drawn_imgA) #img
    cv2.imshow('Lane Filtering', drawn_filter)
    cv2.imshow('Image Source2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break
cap.release()
out.release()
cv2.destroyAllWindows()
