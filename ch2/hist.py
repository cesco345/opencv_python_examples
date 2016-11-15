import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bearings.png')
gray = cv2.imread('bearings.png',0)

equ = cv2.equalizeHist(gray)
res = np.hstack((gray,equ)) 


plt.figure(figsize=(10, 4))


plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original', fontsize=12)

plt.axis('off')



plt.subplot(132)
plt.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Grayscale', fontsize=12)
plt.axis('off')

plt.subplot(133)
plt.imshow(res, cmap=plt.cm.gray, vmin=0, vmax=230)
plt.title('Equalized', fontsize=12)
plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05,
                    right=0.99)
plt.show()

