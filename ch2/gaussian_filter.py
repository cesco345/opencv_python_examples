#!/usr/bin/env python

import sys
from PIL import Image
import scipy
from scipy.ndimage import gaussian_filter, label
from pylab import *



image = Image.open(sys.argv[1]).convert('L')
img = np.array(image)

im_g = gaussian_filter(img, 3)
im_norm = (im_g - im_g.min()) / (float(im_g.max()) - im_g.min())
im_norm[im_norm < 0.5] = 0
im_norm[im_norm >= 0.5] = 1

out = 255 - (im_norm * 255).astype(np.uint8)
print u"\n\n**********  Cells Counted: %d  **********\n\n" % label(out)[1]

print out

figure()
gray()
imshow(out)


show()
