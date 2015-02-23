import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('apple_tag.tiff',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)

while(1):

    cv2.imshow('image',laplacian)
    k = 0xFF & cv2.waitKey(1)

# Exit the program
    if k >0:         # press any key to exit
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

