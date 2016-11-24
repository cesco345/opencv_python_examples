import cv2
import cv2.cv as cv
import numpy as np



img = cv2.imread('yeast2.png',0)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,67,
                            param1=5,param2=1,minRadius=4,maxRadius=20)

circles = np.uint16(np.around(circles))
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)

#cv2.imshow('Detected cells',cimg)
cv2.imshow('detected circles',th2)
cv2.waitKey(0)
cv2.destroyAllWindows()
