import numpy as np
import cv2



img = np.zeros((500,500), dtype=np.float)
img[150:350, 150:350] = 1

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
