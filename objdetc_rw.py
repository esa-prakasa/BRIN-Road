 1. 	Pembuatan Classifier (File XML)

createsamples.exe -info positive/info.txt -vec vector/facevector.vec -num 200 -w 24 -h 24
 
haartraining.exe -data cascades -vec vector/vector.vec -bg negative/bg.txt
-npos 200 -nneg 200 -nstages 15 -mem 1024 -mode ALL -w 24 -h 24 –nonsym
  2. 	Program utama 
import cv2
#import time
#import numpy as np
 
cascade_src = 'cascade.xml'
video_src = 'dataset BRIN Road.avi'
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
 
while True:
	ret, img = cap.read()
  
	if (type(img) == type(None)):
    	break
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cars = car_cascade.detectMultiScale(gray, 1.1, 2)
	for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
	
    cv2.imshow('video', img)
  
	if cv2.waitKey(33) == 27:
    	break
 cv2.destroyAllWindows()
