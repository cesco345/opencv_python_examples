import cv2

img = cv2.imread('yeast2.png',0)

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('Detected Cells',th2)
cv2.waitKey(0)
cv2.destroyAllWindows()
