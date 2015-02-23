import cv2
import numpy as np

img = cv2.imread('apple_tag.png')
gray = cv2.imread('apple_tag.png', 0)
def nothing(x):
	pass
cv2.namedWindow('Image')

cv2.createTrackbar('trackbar1', 'Image', 0, 5, nothing)

threshold = np.zeros(img.shape, np.uint8)
while(1):
	trackbar1 = cv2.getTrackbarPos('trackbar1', 'Image')
	laplacian = cv2.Laplacian(img, trackbar1)
	cv2.imshow('Image', laplacian )
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()
