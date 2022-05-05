import cv2
import numpy as np

plate_cascade = cv2.CascadeClassifier('/home/tsoprano/Documents/UTA/AI_Project/data/cascade.xml')

cap = cv2.VideoCapture('ktmTraff.mp4')

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	plates =  plate_cascade.detectMultiScale(gray, 30, 30)
	for (x,y,w,h) in plates:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		#roi_gray = gray[y:y+h, x:x+w]
		#roi_color = img[y:y+h, x:x+h]
	cv2.imshow('img', img)
	k = cv2.waitKey(5) & 0xff
	if k==27:
		break
		
cap.release()
cv2.destroyAllWindows()
		
