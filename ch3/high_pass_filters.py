import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('apple_tag.png',0)

laplacian = cv2.Laplacian(im,cv2.CV_64F)
sobel_x = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=3)

plt.subplot(2,2,1),plt.imshow(im,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobel_x,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel_y,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
