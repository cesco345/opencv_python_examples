#!/usr/bin/env python
'''
This example shows how to apply benchmarking to an image conversion from rgb to greyscale using a cpu.
I modified code taken from the github notebook found here:
http://nbviewer.ipython.org/github/mroberts3000/GpuComputing/blob/master/IPython/Map.ipynb

Author: Mike Roberts
'''
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import sys
import time
from PIL import Image

image = Image.open("providence.png")
image_array_rgb = np.array(image)

r,g,b = np.split(image_array_rgb, 3, axis=2)
a = np.ones_like(r) * 255

image_array_rgba = np.concatenate((r,g,b,a), axis=2).copy()

if sys.platform == "win32":
    print "Using time.clock for benchmarking...\n" 
    system_timer_function = time.clock
else:
    print "Using time.time for benchmarking...\n" 
    system_timer_function = time.time
    
num_timing_iterations = 100
print "num_timing_iterations = %d" % num_timing_iterations

total_time_seconds = 0
image_array_rgba_uint32 = image_array_rgba.astype(np.uint32)

for i in range(num_timing_iterations):

    start_time_seconds = system_timer_function()

    image_array_greyscale_cpu_result = \
        (image_array_rgba_uint32[:,:,0] + image_array_rgba_uint32[:,:,1] + image_array_rgba_uint32[:,:,2]) / 3

    end_time_seconds = system_timer_function()
    elapsed_time_seconds = (end_time_seconds - start_time_seconds)
    total_time_seconds = total_time_seconds + elapsed_time_seconds

average_time_seconds_cpu = total_time_seconds / num_timing_iterations

print "Using system timer for benchmarking (see above)..."
print "Average time elapsed executing color to greyscale conversion on the CPU over %d runs: %f s" % (num_timing_iterations,average_time_seconds_cpu)

print
rcParams['figure.figsize'] = 6,4

plt.imshow(image_array_greyscale_cpu_result, cmap="gray", vmin=0, vmax=255);
plt.title("image_array_greyscale_cpu_result")
plt.show()
