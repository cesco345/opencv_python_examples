import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread('bearings.png',0)
equ = cv2.equalizeHist(img)
plt.hist(equ,256,[0,256])
plt.show()
