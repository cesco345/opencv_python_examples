#!/usr/bin/env python

from PIL import Image
from pylab import *
import matplotlib.pyplot as plt

im = array(Image.open('newport.png').convert('L'))
im2 = array(Image.open('newport.png'))

plt.figure(figsize=(10, 3.6))


plt.subplot(131)
plt.imshow(im2, cmap=plt.cm.gray)
plt.title('Original', fontsize=12)

plt.axis('off')



plt.subplot(132)
plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Low Contrast', fontsize=12)
plt.axis('off')

plt.subplot(133)
plt.imshow(im, cmap=plt.cm.gray, vmin=100, vmax=230)
plt.title('High Contrast', fontsize=12)
plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05,
                    right=0.99)

figure('Contours')

gray()
contour(im, origin='image')
axis('equal')
axis('off')
show()
