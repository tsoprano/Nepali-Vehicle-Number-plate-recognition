import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
plate_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')
car_cascade = cv2.CascadeClassifier('cas5.xml')

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('ktmTraff.mp4')
lower_cyan = np.array([80,70,50])
upper_cyan = np.array([100,255,255])
framewidth = 640
frameheight = 480

def empty(a):
    pass

cv2.namedWindow('Result')
cv2.resizeWindow('Result', framewidth, frameheight+100)
cv2.createTrackbar('Scale', 'Result', 400, 1000, empty)
cv2.createTrackbar('Neigh', 'Result', 8, 20, empty)
cv2.createTrackbar('Min Area', 'Result', 0, 100000, empty)

img = cv2.imread('222.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cars = car_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in cars:
    # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    # cv2.imshow('img', img)
    # k = cv2.waitKey(30000) & 0xff
    # if k==27:
        # break

 		# roi_gray = gray[y:y+h, x:x+w]
 		# roi_color = img[y:y+h, x:x+w]
 		# inv_img = cv2.bitwise_not(roi_color)
 		# hsv_inv = cv2.cvtColor(inv_img, cv2.COLOR_BGR2HSV)
 		# mask2 = cv2.inRange(hsv_inv, lower_cyan, upper_cyan)
 		# res2 = cv2.bitwise_and(roi_color, roi_color, mask=mask2)
        # k = cv2.waitKey(30) & 0xff
        # if k==27:
            # break


while True:
    ret, img = cap2.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaleval = 1+ (cv2.getTrackbarPos('Scale', 'Result')/1000)
    neigh = cv2.getTrackbarPos('Neigh', 'Result')
    cars = car_cascade.detectMultiScale(gray, scaleval, neigh)
    for (x,y,w,h) in cars:
        area = w*h
        minArea = cv2.getTrackbarPos('Min Area', 'Result')
        if area>minArea:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_color = img[y:y+h, x:x+w]
            
        # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # inv_img = cv2.bitwise_not(roi_color)
        # hsv_inv = cv2.cvtColor(inv_img, cv2.COLOR_BGR2HSV)
        # mask2 = cv2.inRange(hsv_inv, lower_cyan, upper_cyan)
        # res2 = cv2.bitwise_and(roi_color, roi_color, mask=mask2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break


#while True:
#	ret, img = cap.read()
#	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#	for(x,y,w,h) in faces:
#		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
#		roi_gray = gray[y:y+h, x:x+w]
#		roi_color = img[y:y+h, x:x+w]
#		eyes = eye_cascade.detectMultiScale(roi_gray)
#		for (ex,ey,ew,eh) in eyes:
#			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

#	cv2.imshow('img', img)
#	k = cv2.waitKey(30) & 0xff
#	if k==27:
#		break


# cap.release()
# cv2.destroyAllWindows()