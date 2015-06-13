import cv2
import numpy as np

img = cv2.imread('plate.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


filtered = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,89,3)

kernel = np.ones((5,5), np.uint8)
open = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

cv2.imshow('img',close)
cv2.waitKey(0)
cv2.destroyAllWindows()
