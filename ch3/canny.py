#!/usr/bin/env python

import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Image')

img = cv2.imread('apple_tag.png', 0)     

cv2.createTrackbar('trackbar1', 'Image', 0, 255, nothing)
cv2.createTrackbar('trackbar2', 'Image', 0, 255, nothing)

while(1):
    trackbar1 = cv2.getTrackbarPos('trackbar1', 'Image')
    trackbar2 = cv2.getTrackbarPos('trackbar2', 'Image')
    edges = cv2.Canny(img, trackbar1, trackbar2)
    cv2.imshow('Image',edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
