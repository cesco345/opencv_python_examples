#!/usr/bin/env python
'''
This example shows how to apply benchmarking to an image conversion using a cpu and a gpu.
I modified code taken from the github notebook found here:
http://nbviewer.ipython.org/github/mroberts3000/GpuComputing/blob/master/IPython/Map.ipynb

Author: Mike Roberts
'''
import numpy
from matplotlib import pyplot
from pylab import *
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler
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
image_array_greyscale_gpu_result = np.zeros_like(a).squeeze()

source_module = pycuda.compiler.SourceModule \
(
"""
__global__ void color_to_greyscale(
    unsigned char* d_greyscale,
    uchar4* d_color,
    int num_pixels_y,
    int num_pixels_x )
{
    int ny = num_pixels_y;
    int  nx = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        uchar4 color = d_color[ image_index_1d ];
        unsigned int greyscale = ( color.x + color.y + color.z ) / 3;
    
        d_greyscale[ image_index_1d ] = (unsigned char)greyscale;
    }
}
"""
)

color_to_greyscale_function = source_module.get_function("color_to_greyscale")

start_timer = pycuda.driver.Event()
end_timer = pycuda.driver.Event()

num_pixels_y = np.int32(image_array_rgba.shape[0])
num_pixels_x = np.int32(image_array_rgba.shape[1])
color_to_greyscale_function_block = (32,16,1)

num_blocks_y = int(ceil(float(num_pixels_y) / float(color_to_greyscale_function_block[1])))
num_blocks_x = int(ceil(float(num_pixels_x) / float(color_to_greyscale_function_block[0])))
color_to_greyscale_function_grid  = (num_blocks_x, num_blocks_y)

image_array_rgba_device = pycuda.driver.mem_alloc(image_array_rgba.nbytes)
image_array_greyscale_device = pycuda.driver.mem_alloc(image_array_greyscale_gpu_result.nbytes)

pycuda.driver.memcpy_htod(image_array_rgba_device, image_array_rgba)
pycuda.driver.memcpy_htod(image_array_greyscale_device, image_array_greyscale_gpu_result)

color_to_greyscale_function(
    image_array_greyscale_device,
    image_array_rgba_device,
    num_pixels_y,
    num_pixels_x,
    block=color_to_greyscale_function_block,
    grid=color_to_greyscale_function_grid)

pycuda.driver.memcpy_dtoh(image_array_greyscale_gpu_result, image_array_greyscale_device)


for i in range(num_timing_iterations):
    pycuda.driver.memcpy_htod(image_array_greyscale_device, image_array_greyscale_gpu_result)
    pycuda.driver.Context.synchronize()
    start_time_seconds = system_timer_function()
    color_to_greyscale_function(
        image_array_greyscale_device,
        image_array_rgba_device,
        num_pixels_y,
        num_pixels_x,
        block=color_to_greyscale_function_block,
        grid=color_to_greyscale_function_grid)

    pycuda.driver.Context.synchronize()
    
    end_time_seconds = system_timer_function()
    elapsed_time_seconds = end_time_seconds - start_time_seconds
    total_time_seconds = total_time_seconds + elapsed_time_seconds

pycuda.driver.memcpy_dtoh(image_array_greyscale_gpu_result, image_array_greyscale_device)

average_time_seconds_gpu = total_time_seconds / num_timing_iterations


total_time_seconds  = 0
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
print "Average time elapsed executing color_to_greyscale GPU kernel over %d runs: %f s" % (num_timing_iterations,average_time_seconds_gpu)

print

rcParams['figure.figsize'] = 6,4

plt.imshow(image_array_greyscale_gpu_result, cmap="gray", vmin=0, vmax=255);
plt.title("image_array_greyscale_gpu_result");
plt.show()

gpu_speedup = average_time_seconds_cpu / average_time_seconds_gpu

print "Average CPU time: %f s" % average_time_seconds_cpu
print "Average GPU time: %f s" % average_time_seconds_gpu
print "GPU speedup:      %f x" % gpu_speedup
