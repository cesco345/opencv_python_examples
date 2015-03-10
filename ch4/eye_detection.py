#!/usr/bin/env python
'''
This example shows how to locate and identify a template image within
a source image. It uses the matchTemplate function and examples found in the 
OpenCV python tutorials documentation page.

I also used and modified code taken from the github notebook found here:
http://nbviewer.ipython.org/github/mroberts3000/GpuComputing/blob/master/IPython/Map.ipynb

Author: Mike Roberts
'''

import PIL
import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import cv2

# load source image and template image with opencv, and conversion to greyscale
source = cv2.imread('max.png')
img_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
template = cv2.imread('max_eye.png',0)
w, h = template.shape[::-1]

# splitting into color channels from rgb-a to bgr-a for easier plotting
# of the bgra image
r,g,b = np.split(source, 3, axis=2)
a = np.ones_like(r) * 255
bgra = np.concatenate((b,g,r,a), axis=2).copy()

# perform normalized cross-correlation with the tm.ccoeff function and threshold
# to locate the template image in the source image
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# perform normalized cross-correlation for template matching
image                            = PIL.Image.open("max.png")
image_template                   = PIL.Image.open("max_eye.png")

image_array_rgb                  = np.array(image)
image_template_array_rgb         = np.array(image_template)

r,g,b                            = np.split(image_array_rgb, 3, axis=2)
r_template,g_template,b_template = np.split(image_template_array_rgb, 3, axis=2)

a                                = np.ones_like(r) * 255
a_template                       = np.ones_like(r_template) * 255

rgba                             = np.concatenate((r,g,b,a), axis=2).copy()
rgba_template                    = np.concatenate((r_template,g_template,b_template,a_template), axis=2).copy()

r = r.squeeze().copy()
g = g.squeeze().copy()
b = b.squeeze().copy()
a = a.squeeze().copy()

r_template = r_template.squeeze().copy()
g_template = g_template.squeeze().copy()
b_template = b_template.squeeze().copy()

r_response_opencv = cv2.matchTemplate(r.astype(np.float32), r_template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
g_response_opencv = cv2.matchTemplate(g.astype(np.float32), g_template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
b_response_opencv = cv2.matchTemplate(b.astype(np.float32), b_template.astype(np.float32), cv2.TM_CCOEFF_NORMED)

response_opencv   = r_response_opencv * g_response_opencv * b_response_opencv


rcParams['figure.figsize'] = 20,4

plt.subplot(141),plt.imshow(bgra,cmap = 'gray')
plt.title('Red Eye Source'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(res,cmap = 'gray')
plt.title('Resulting Match'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(img_gray,cmap = 'gray')
plt.title('Template Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(response_opencv > 0.24, cmap="gray");
plt.title("cv_res > 0.24"), plt.xticks([]), plt.yticks([])
plt.show()
