import cv2
import cv2.cv as cv
import numpy as np



img = cv2.imread('yeast2.png',0)

c_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,67,
                            param1=5,param2=1,minRadius=4,maxRadius=20)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(c_img,(i[0],i[1]),i[2],(0,0,0),2)

cv2.imshow('Detected Cells',c_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
