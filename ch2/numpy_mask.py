#!/usr/bin/env python 

import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt



img2 = np.zeros((500,500), np.int8)
random = np.random.rand(500, 500)
img = cv2.imread('white.png')

ret,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY)

plt.subplot(131),plt.imshow(thresh1,'gray'),plt.title('White-255')
plt.xticks([]),plt.yticks([])

plt.subplot(132),plt.imshow(img2,'gray'),plt.title('Mask-0')
plt.xticks([]),plt.yticks([])

plt.subplot(133),plt.imshow(random,'gray'),plt.title('Random')
plt.xticks([]),plt.yticks([])

plt.show()
