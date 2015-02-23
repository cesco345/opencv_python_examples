import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('apple_tag.png')
gray_img = cv2.imread('apple_tag.png',0)
blur = cv2.medianBlur(gray_img,5)

ret,th1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
images = [img, 0, th1,
          gray_img, 0, th2,
          blur, 0, th3]
titles = ['Original Image','Histogram','GlobalThresholding',
          'Original Gray','Histogram',"Adaptive Mean Thresholding",
          'Gaussian Blur','Histogram',"Adaptive Gaussian Thresholding"]
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)

    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
