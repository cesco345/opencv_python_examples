#!/usr/bin/env python
import cv2 

image = cv2.imread("newport.png") 
cv2.imshow("original", image) 


sliced = image[240:380, 550:742] 

cv2.imshow("Sliced", sliced) 

cv2.waitKey(0) 
cv2.destroyAllWindows()
