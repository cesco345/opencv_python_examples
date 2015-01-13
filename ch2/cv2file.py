#!/usr/bin/env python

import cv2 

img = cv2.imread('beach.png') 

print type(img) 
print 'img shape: ', img.shape 
print 'img.dtype: ', img.dtype 
print 'img.size: ', img.size 

cv2.imshow('Original', img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
