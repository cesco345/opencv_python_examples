#!/usr/bin/env python 

import numpy as np
import cv2
from matplotlib import pyplot as plt


img2 = np.zeros((500,500), np.int8)
random = np.random.rand(500, 500)

plt.subplot(131),plt.imshow(img2,'gray'),plt.title('Mask-0')
plt.xticks([]),plt.yticks([])

plt.subplot(133),plt.imshow(random,'gray'),plt.title('Random')
plt.xticks([]),plt.yticks([])

plt.show()
