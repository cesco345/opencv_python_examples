import cv2
import numpy as np

img = cv2.imread('plate.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(gray)    # Remember histogram equalization works only for grayscale images

cv2.imshow('src',gray)
cv2.imshow('equ',equ)
cv2.waitKey(0)
cv2.destroyAllWindows()
