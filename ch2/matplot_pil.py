#using matplotlib

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2

original = mpimg.imread('home_tree.png')

gray = cv2.imread('home_tree.png',0)

plt.subplot(131),plt.imshow(gray, cmap = 'gray'),plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(original, cmap='gray'),plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.show()
