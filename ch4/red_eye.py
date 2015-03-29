#!/usr/bin/env python

import PIL
import PIL.Image
import numpy
from matplotlib import pyplot as plt
from pylab import *
import cv2
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler
import radixsort

image                            = PIL.Image.open("max.png")
image_template                   = PIL.Image.open("max_eye.png")

image_array_rgb                  = numpy.array(image)
image_template_array_rgb         = numpy.array(image_template)

r,g,b                            = numpy.split(image_array_rgb, 3, axis=2)
r_template,g_template,b_template = numpy.split(image_template_array_rgb, 3, axis=2)

a                                = numpy.ones_like(r) * 255
a_template                       = numpy.ones_like(r_template) * 255

rgba                             = numpy.concatenate((r,g,b,a), axis=2).copy()
rgba_template                    = numpy.concatenate((r_template,g_template,b_template,a_template), axis=2).copy()

r = r.squeeze().copy()
g = g.squeeze().copy()
b = b.squeeze().copy()
a = a.squeeze().copy()

r_template = r_template.squeeze().copy()
g_template = g_template.squeeze().copy()
b_template = b_template.squeeze().copy()


rcParams['figure.figsize'] = 9,4
#figsize(9,4)

plt.subplot(121)
plt.imshow(rgba);
plt.title("rgba");

plt.subplot(122)
plt.imshow(rgba_template, interpolation="nearest");
plt.title("rgba_template");

plt.show()


r_response_opencv = cv2.matchTemplate(r.astype(numpy.float32), r_template.astype(numpy.float32), cv2.TM_CCOEFF_NORMED)
g_response_opencv = cv2.matchTemplate(g.astype(numpy.float32), g_template.astype(numpy.float32), cv2.TM_CCOEFF_NORMED)
b_response_opencv = cv2.matchTemplate(b.astype(numpy.float32), b_template.astype(numpy.float32), cv2.TM_CCOEFF_NORMED)

response_opencv   = r_response_opencv * g_response_opencv * b_response_opencv


rcParams['figure.figsize'] = 20,4
#figsize(20,4)

plt.subplot(141);
plt.imshow(r_response_opencv, cmap="gray");
plt.title("r_response_opencv");
plt.colorbar();

plt.subplot(142);
plt.imshow(g_response_opencv, cmap="gray");
plt.title("g_response_opencv");
plt.colorbar();

plt.subplot(143);
plt.imshow(b_response_opencv, cmap="gray");
plt.title("b_response_opencv");
plt.colorbar();

plt.subplot(144);
plt.imshow(response_opencv, cmap="gray");
plt.title("response_opencv");
plt.colorbar();
plt.show()


rcParams['figure.figsize'] = 4,4
#figsize(4,4)

plt.imshow(response_opencv > 0.15, cmap="gray");
plt.title("response_opencv > 0.15");
plt.colorbar();
plt.show()


num_pixels_y                 = numpy.int32(r.shape[0])
num_pixels_x                 = numpy.int32(r.shape[1])

r_response_gpu_naive         = numpy.zeros_like(r, dtype=numpy.float32)
g_response_gpu_naive         = numpy.zeros_like(g, dtype=numpy.float32)
b_response_gpu_naive         = numpy.zeros_like(b, dtype=numpy.float32)
response_gpu_naive           = numpy.zeros_like(r, dtype=numpy.float32)

r_response_gpu_shared_memory = numpy.zeros_like(r, dtype=numpy.float32)
g_response_gpu_shared_memory = numpy.zeros_like(g, dtype=numpy.float32)
b_response_gpu_shared_memory = numpy.zeros_like(b, dtype=numpy.float32)
response_gpu_shared_memory   = numpy.zeros_like(r, dtype=numpy.float32)

positive_response_gpu        = numpy.zeros_like(r, dtype=numpy.float32)
coordinates_gpu              = numpy.zeros_like(r, dtype=numpy.uint32)
sorted_positive_response_gpu = numpy.zeros_like(r, dtype=numpy.float32)
sorted_coordinates_gpu       = numpy.zeros_like(r, dtype=numpy.uint32)

r_output_gpu                 = numpy.zeros_like(r)

r_device                     = pycuda.driver.mem_alloc(r.nbytes)
g_device                     = pycuda.driver.mem_alloc(g.nbytes)
b_device                     = pycuda.driver.mem_alloc(b.nbytes)

r_template_device            = pycuda.driver.mem_alloc(r_template.nbytes)
g_template_device            = pycuda.driver.mem_alloc(g_template.nbytes)
b_template_device            = pycuda.driver.mem_alloc(b_template.nbytes)

r_response_device            = pycuda.driver.mem_alloc(r_response_gpu_naive.nbytes)
g_response_device            = pycuda.driver.mem_alloc(g_response_gpu_naive.nbytes)
b_response_device            = pycuda.driver.mem_alloc(b_response_gpu_naive.nbytes)
response_device              = pycuda.driver.mem_alloc(response_gpu_naive.nbytes)

positive_response_device        = pycuda.driver.mem_alloc(positive_response_gpu.nbytes)
coordinates_device              = pycuda.driver.mem_alloc(coordinates_gpu.nbytes)
sorted_positive_response_device = pycuda.driver.mem_alloc(sorted_positive_response_gpu.nbytes)
sorted_coordinates_device       = pycuda.driver.mem_alloc(sorted_coordinates_gpu.nbytes)

r_output_device              = pycuda.driver.mem_alloc(r.nbytes)

radix_sort_manager = radixsort.RadixSortManager(512*512)

source_module = pycuda.compiler.SourceModule \
(
"""
__global__ void naive_normalized_cross_correlation(
    float*         d_response,
    unsigned char* d_original,
    unsigned char* d_template,
    int            num_pixels_y,
    int            num_pixels_x,
    int            template_half_height,
    int            template_height,
    int            template_half_width,
    int            template_width,
    int            template_size,
    float          template_mean
)
{
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int  knx            = template_width;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        //
        // compute image mean
        //
        float image_sum = 0.0f;

        for ( int y = -template_half_height; y <= template_half_height; y++ )
        {
            for ( int x = -template_half_width; x <= template_half_width; x++ )
            {
                int2 image_offset_index_2d         =
                    make_int2( image_index_2d.x + x, image_index_2d.y + y );
                int2 image_offset_index_2d_clamped =
                    make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
                int  image_offset_index_1d_clamped =
                    ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
                
                unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];

                image_sum += (float)image_offset_value;
            }
        }

        float image_mean = image_sum / (float)template_size;

        //
        // compute sums
        //
        float sum_of_image_template_diff_products = 0.0f;
        float sum_of_squared_image_diffs          = 0.0f;
        float sum_of_squared_template_diffs       = 0.0f;

        for ( int y = -template_half_height; y <= template_half_height; y++ )
        {
            for ( int x = -template_half_width; x <= template_half_width; x++ )
            {
                int2 image_offset_index_2d         =
                    make_int2( image_index_2d.x + x, image_index_2d.y + y );
                int2 image_offset_index_2d_clamped =
                    make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
                int  image_offset_index_1d_clamped =
                    ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
                
                unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];
                float         image_diff         = (float)image_offset_value - image_mean;

                int2 template_index_2d = make_int2( x + template_half_width, y + template_half_height );
                int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

                unsigned char template_value = d_template[ template_index_1d ];
                float         template_diff  = template_value - template_mean;

                float image_template_diff_product = image_offset_value   * template_diff;
                float squared_image_diff          = image_diff           * image_diff;
                float squared_template_diff       = template_diff        * template_diff;

                sum_of_image_template_diff_products += image_template_diff_product;
                sum_of_squared_image_diffs          += squared_image_diff;
                sum_of_squared_template_diffs       += squared_template_diff;
            }
        }
        
        //
        // compute final result
        //
        float result_value = 0.0f;

        if ( sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0 )
        {
            result_value = sum_of_image_template_diff_products / sqrt( sum_of_squared_image_diffs * sum_of_squared_template_diffs );
        }

        d_response[ image_index_1d ] = result_value;
    }
}

__global__ void combined_response(
    float*         d_response,
    float*         d_r_response,
    float*         d_g_response,
    float*         d_b_response,
    int            num_pixels_y,
    int            num_pixels_x
)
{
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        d_response[ image_index_1d ] = d_r_response[ image_index_1d ] * d_g_response[ image_index_1d ] * d_b_response[ image_index_1d ];
    }
}
"""
)

naive_normalized_cross_correlation_function       = source_module.get_function("naive_normalized_cross_correlation")

template_height                                   = numpy.int32(r_template.shape[0])
template_width                                    = numpy.int32(r_template.shape[1])

template_half_height                              = numpy.int32((template_height - 1) / 2)
template_half_width                               = numpy.int32((template_width  - 1) / 2)

template_size                                     = numpy.int32(template_height * template_width)
template_mean_r                                   = numpy.float32(numpy.mean(r_template))
template_mean_g                                   = numpy.float32(numpy.mean(g_template))
template_mean_b                                   = numpy.float32(numpy.mean(b_template))

naive_normalized_cross_correlation_function_block = (16,16,1)
num_blocks_y                                      = int(ceil(float(num_pixels_y) / float(naive_normalized_cross_correlation_function_block[1])))
num_blocks_x                                      = int(ceil(float(num_pixels_x) / float(naive_normalized_cross_correlation_function_block[0])))
naive_normalized_cross_correlation_function_grid  = (num_blocks_x, num_blocks_y)

combined_response_function                        = source_module.get_function("combined_response")

combined_response_function_block                  = (32,16,1)
num_blocks_y                                      = int(ceil(float(num_pixels_y) / float(combined_response_function_block[1])))
num_blocks_x                                      = int(ceil(float(num_pixels_x) / float(combined_response_function_block[0])))
combined_response_function_grid                   = (num_blocks_x, num_blocks_y)


pycuda.driver.memcpy_htod(r_device,                    r)
pycuda.driver.memcpy_htod(g_device,                    g)
pycuda.driver.memcpy_htod(b_device,                    b)

pycuda.driver.memcpy_htod(r_response_device,           r_response_gpu_naive)
pycuda.driver.memcpy_htod(g_response_device,           g_response_gpu_naive)
pycuda.driver.memcpy_htod(b_response_device,           b_response_gpu_naive)
pycuda.driver.memcpy_htod(response_device,             response_gpu_naive)

pycuda.driver.memcpy_htod(r_template_device,           r_template)
pycuda.driver.memcpy_htod(g_template_device,           g_template)
pycuda.driver.memcpy_htod(b_template_device,           b_template)

pycuda.driver.Context.synchronize()

naive_normalized_cross_correlation_function(
    r_response_device,
    r_device,
    r_template_device,
    num_pixels_y,
    num_pixels_x,
    template_half_height,
    template_height,
    template_half_width,
    template_width,
    template_size,
    template_mean_r,
    block=naive_normalized_cross_correlation_function_block,
    grid=naive_normalized_cross_correlation_function_grid)

pycuda.driver.Context.synchronize()

naive_normalized_cross_correlation_function(
    g_response_device,
    g_device,
    g_template_device,
    num_pixels_y,
    num_pixels_x,
    template_half_height,
    template_height,
    template_half_width,
    template_width,
    template_size,
    template_mean_g,
    block=naive_normalized_cross_correlation_function_block,
    grid=naive_normalized_cross_correlation_function_grid)

pycuda.driver.Context.synchronize()

naive_normalized_cross_correlation_function(
    b_response_device,
    b_device,
    b_template_device,
    num_pixels_y,
    num_pixels_x,
    template_half_height,
    template_height,
    template_half_width,
    template_width,
    template_size,
    template_mean_b,
    block=naive_normalized_cross_correlation_function_block,
    grid=naive_normalized_cross_correlation_function_grid)

pycuda.driver.Context.synchronize()

combined_response_function(
    response_device,
    r_response_device,
    g_response_device,
    b_response_device,
    num_pixels_y,
    num_pixels_x,
    block=combined_response_function_block,
    grid=combined_response_function_grid)

pycuda.driver.Context.synchronize()

pycuda.driver.memcpy_dtoh(r_response_gpu_naive, r_response_device)
pycuda.driver.memcpy_dtoh(g_response_gpu_naive, g_response_device)
pycuda.driver.memcpy_dtoh(b_response_gpu_naive, b_response_device)
pycuda.driver.memcpy_dtoh(response_gpu_naive,   response_device)


rcParams['figure.figsize'] = 20,4
#figsize(20,4)

plt.subplot(141);
plt.imshow(r_response_gpu_naive, cmap="gray");
plt.title("r_response_gpu_naive");
plt.colorbar();

plt.subplot(142);
plt.imshow(g_response_gpu_naive, cmap="gray");
plt.title("g_response_gpu_naive");
plt.colorbar();

plt.subplot(143);
plt.imshow(b_response_gpu_naive, cmap="gray");
plt.title("b_response_gpu_naive");
plt.colorbar();

plt.subplot(144);
plt.imshow(response_gpu_naive, cmap="gray");
plt.title("response_gpu_naive");
plt.colorbar();
plt.show()



source_module = pycuda.compiler.SourceModule \
(
"""
#define BLOCK_SIZE_Y           16
#define BLOCK_SIZE_X           16
#define TEMPLATE_HALF_HEIGHT   16
#define TEMPLATE_HALF_WIDTH    16
#define SHARED_MEMORY_SIZE_Y   BLOCK_SIZE_Y + ( 2 * TEMPLATE_HALF_HEIGHT )
#define SHARED_MEMORY_SIZE_X   BLOCK_SIZE_X + ( 2 * TEMPLATE_HALF_WIDTH ) + 1
#define SHARED_MEMORY_OFFSET_Y TEMPLATE_HALF_HEIGHT
#define SHARED_MEMORY_OFFSET_X TEMPLATE_HALF_WIDTH

__global__ void shared_memory_normalized_cross_correlation(
    float*         d_response,
    unsigned char* d_original,
    unsigned char* d_template,
    int            num_pixels_y,
    int            num_pixels_x,
    int            template_half_height,
    int            template_height,
    int            template_half_width,
    int            template_width,
    int            template_size,
    float          template_mean
)
{
    __shared__ unsigned char s_original[ SHARED_MEMORY_SIZE_Y ][ SHARED_MEMORY_SIZE_X ];

    int  ny                            = num_pixels_y;
    int  nx                            = num_pixels_x;
    int  knx                           = template_width;
    int2 image_index_2d_global         =
        make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int2 image_index_2d_global_clamped =
        make_int2( min( nx - 1, max( 0, image_index_2d_global.x ) ), min( ny - 1, max( 0, image_index_2d_global.y ) ) );
    int  image_index_1d_global_clamped =
        ( nx * image_index_2d_global_clamped.y ) + image_index_2d_global_clamped.x;
    int2 image_index_2d_shared_memory  =
        make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y );

    //
    // load center of shared memory
    //
    s_original[ image_index_2d_shared_memory.y ][ image_index_2d_shared_memory.x ] = d_original[ image_index_1d_global_clamped ];
    
    //
    // load y+1 halo into shared memory
    //
    if ( threadIdx.y < TEMPLATE_HALF_HEIGHT )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + 1 ) * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y + BLOCK_SIZE_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] = d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load y-1 halo into shared memory
    //
    if ( threadIdx.y >= BLOCK_SIZE_Y - TEMPLATE_HALF_HEIGHT )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( ( blockIdx.y - 1 ) * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y - BLOCK_SIZE_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load x+1 halo into shared memory
    //
    if ( threadIdx.x < TEMPLATE_HALF_WIDTH )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( ( blockIdx.x + 1 ) * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X + BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load x-1 halo into shared memory
    //
    if ( threadIdx.x >= BLOCK_SIZE_X - TEMPLATE_HALF_WIDTH )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( ( blockIdx.x - 1 ) * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X - BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load x+1,y+1 halo into shared memory
    //
    if ( threadIdx.x < TEMPLATE_HALF_WIDTH && threadIdx.y < TEMPLATE_HALF_HEIGHT )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( ( blockIdx.x + 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + 1 ) * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X + BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y + BLOCK_SIZE_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load x+1,y-1 halo into shared memory
    //
    if ( threadIdx.x < TEMPLATE_HALF_WIDTH && threadIdx.y >= BLOCK_SIZE_Y - TEMPLATE_HALF_HEIGHT )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( ( blockIdx.x + 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y - 1 ) * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X + BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y - BLOCK_SIZE_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load x-1,y+1 halo into shared memory
    //
    if ( threadIdx.x >= BLOCK_SIZE_X - TEMPLATE_HALF_WIDTH && threadIdx.y < TEMPLATE_HALF_HEIGHT )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( ( blockIdx.x - 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y + 1 ) * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X - BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y + BLOCK_SIZE_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // load x-1,y-1 halo into shared memory
    //
    if ( threadIdx.x >= BLOCK_SIZE_X - TEMPLATE_HALF_WIDTH && threadIdx.y >= BLOCK_SIZE_Y - TEMPLATE_HALF_HEIGHT )
    {
        int2 image_halo_index_2d_global         =
            make_int2( ( ( blockIdx.x - 1 ) * blockDim.x ) + threadIdx.x, ( ( blockIdx.y - 1 ) * blockDim.y ) + threadIdx.y );
        int2 image_halo_index_2d_global_clamped =
            make_int2( min( nx - 1, max( 0, image_halo_index_2d_global.x ) ), min( ny - 1, max( 0, image_halo_index_2d_global.y ) ) );
        int  image_halo_index_1d_global_clamped =
            ( nx * image_halo_index_2d_global_clamped.y ) + image_halo_index_2d_global_clamped.x;
        int2 image_halo_index_2d_shared_memory  =
            make_int2( threadIdx.x + SHARED_MEMORY_OFFSET_X - BLOCK_SIZE_X, threadIdx.y + SHARED_MEMORY_OFFSET_Y - BLOCK_SIZE_Y );

        s_original[ image_halo_index_2d_shared_memory.y ][ image_halo_index_2d_shared_memory.x ] =
            d_original[ image_halo_index_1d_global_clamped ];
    }

    //
    // wait until all threads in the thread block are finished loading the image chunk into shared memory
    //
    __syncthreads();

    if ( image_index_2d_global.x < nx && image_index_2d_global.y < ny )
    {
        //
        // compute image mean
        //
        float image_sum = 0.0f;

        for ( int y = -template_half_height; y <= template_half_height; y++ )
        {
            for ( int x = -template_half_width; x <= template_half_width; x++ )
            {
                int2          image_offset_index_2d      =
                    make_int2( image_index_2d_shared_memory.x + x, image_index_2d_shared_memory.y + y );
                unsigned char image_offset_value         =
                    s_original[ image_offset_index_2d.y ][ image_offset_index_2d.x ];

                image_sum += (float)image_offset_value;
            }
        }

        float image_mean = image_sum / (float)template_size;

        //
        // compute sums
        //
        float sum_of_image_template_diff_products = 0.0f;
        float sum_of_squared_image_diffs          = 0.0f;
        float sum_of_squared_template_diffs       = 0.0f;

        for ( int y = -template_half_height; y <= template_half_height; y++ )
        {
            for ( int x = -template_half_width; x <= template_half_width; x++ )
            {
                int2          image_offset_index_2d =
                    make_int2( image_index_2d_shared_memory.x + x, image_index_2d_shared_memory.y + y );

                unsigned char image_offset_value    = s_original[ image_offset_index_2d.y ][ image_offset_index_2d.x ];
                float         image_diff            = (float)image_offset_value - image_mean;

                int2 template_index_2d = make_int2( x + template_half_width, y + template_half_height );
                int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

                unsigned char template_value = d_template[ template_index_1d ];
                float         template_diff  = template_value - template_mean;

                float image_template_diff_product = image_offset_value   * template_diff;
                float squared_image_diff          = image_diff           * image_diff;
                float squared_template_diff       = template_diff        * template_diff;

                sum_of_image_template_diff_products += image_template_diff_product;
                sum_of_squared_image_diffs          += squared_image_diff;
                sum_of_squared_template_diffs       += squared_template_diff;
            }
        }

        //
        // compute final result
        //
        float result_value = 0.0f;

        if ( sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0 )
        {
            result_value = sum_of_image_template_diff_products / sqrt( sum_of_squared_image_diffs * sum_of_squared_template_diffs );
        }

        d_response[ image_index_1d_global_clamped ] = result_value;
    }
}

__global__ void combined_response(
    float*         d_response,
    float*         d_r_response,
    float*         d_g_response,
    float*         d_b_response,
    int            num_pixels_y,
    int            num_pixels_x
)
{
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        d_response[ image_index_1d ] = d_r_response[ image_index_1d ] * d_g_response[ image_index_1d ] * d_b_response[ image_index_1d ];
    }
}
"""
)

shared_memory_normalized_cross_correlation_function       = source_module.get_function("shared_memory_normalized_cross_correlation")

template_height                                           = numpy.int32(r_template.shape[0])
template_width                                            = numpy.int32(r_template.shape[1])

template_half_height                                      = numpy.int32((template_height - 1) / 2)
template_half_width                                       = numpy.int32((template_width  - 1) / 2)

template_size                                             = numpy.int32(template_height * template_width)
template_mean_r                                           = numpy.float32(numpy.mean(r_template))
template_mean_g                                           = numpy.float32(numpy.mean(g_template))
template_mean_b                                           = numpy.float32(numpy.mean(b_template))

shared_memory_normalized_cross_correlation_function_block = (16,16,1)
num_blocks_y                                              = \
    int(ceil(float(num_pixels_y) / float(shared_memory_normalized_cross_correlation_function_block[1])))
num_blocks_x                                              = \
    int(ceil(float(num_pixels_x) / float(shared_memory_normalized_cross_correlation_function_block[0])))
shared_memory_normalized_cross_correlation_function_grid  = (num_blocks_x, num_blocks_y)

combined_response_function                                = source_module.get_function("combined_response")

combined_response_function_block                          = (32,16,1)
num_blocks_y                                              = int(ceil(float(num_pixels_y) / float(combined_response_function_block[1])))
num_blocks_x                                              = int(ceil(float(num_pixels_x) / float(combined_response_function_block[0])))
combined_response_function_grid                           = (num_blocks_x, num_blocks_y)

pycuda.driver.memcpy_htod(r_device,                    r)
pycuda.driver.memcpy_htod(g_device,                    g)
pycuda.driver.memcpy_htod(b_device,                    b)

pycuda.driver.memcpy_htod(r_response_device,           r_response_gpu_shared_memory)
pycuda.driver.memcpy_htod(g_response_device,           g_response_gpu_shared_memory)
pycuda.driver.memcpy_htod(b_response_device,           b_response_gpu_shared_memory)
pycuda.driver.memcpy_htod(response_device,             response_gpu_shared_memory)

pycuda.driver.memcpy_htod(r_template_device,           r_template)
pycuda.driver.memcpy_htod(g_template_device,           g_template)
pycuda.driver.memcpy_htod(b_template_device,           b_template)

pycuda.driver.Context.synchronize()

shared_memory_normalized_cross_correlation_function(
    r_response_device,
    r_device,
    r_template_device,
    num_pixels_y,
    num_pixels_x,
    template_half_height,
    template_height,
    template_half_width,
    template_width,
    template_size,
    template_mean_r,
    block=naive_normalized_cross_correlation_function_block,
    grid=naive_normalized_cross_correlation_function_grid)

pycuda.driver.Context.synchronize()

shared_memory_normalized_cross_correlation_function(
    g_response_device,
    g_device,
    g_template_device,
    num_pixels_y,
    num_pixels_x,
    template_half_height,
    template_height,
    template_half_width,
    template_width,
    template_size,
    template_mean_g,
    block=naive_normalized_cross_correlation_function_block,
    grid=naive_normalized_cross_correlation_function_grid)

pycuda.driver.Context.synchronize()

shared_memory_normalized_cross_correlation_function(
    b_response_device,
    b_device,
    b_template_device,
    num_pixels_y,
    num_pixels_x,
    template_half_height,
    template_height,
    template_half_width,
    template_width,
    template_size,
    template_mean_b,
    block=naive_normalized_cross_correlation_function_block,
    grid=naive_normalized_cross_correlation_function_grid)

pycuda.driver.Context.synchronize()

combined_response_function(
    response_device,
    r_response_device,
    g_response_device,
    b_response_device,
    num_pixels_y,
    num_pixels_x,
    block=combined_response_function_block,
    grid=combined_response_function_grid)

pycuda.driver.Context.synchronize()

pycuda.driver.memcpy_dtoh(r_response_gpu_shared_memory, g_response_device)
pycuda.driver.memcpy_dtoh(g_response_gpu_shared_memory, g_response_device)
pycuda.driver.memcpy_dtoh(b_response_gpu_shared_memory, b_response_device)
pycuda.driver.memcpy_dtoh(response_gpu_shared_memory,   response_device)


rcParams['figure.figsize'] = 20,4
#figsize(20,4)

plt.subplot(141);
plt.imshow(r_response_gpu_shared_memory, cmap="gray");
plt.title("r_response_gpu_shared_memory");
plt.colorbar();

plt.subplot(142);
plt.imshow(g_response_gpu_shared_memory, cmap="gray");
plt.title("g_response_gpu_shared_memory");
plt.colorbar();

plt.subplot(143);
plt.imshow(b_response_gpu_shared_memory, cmap="gray");
plt.title("b_response_gpu_shared_memory");
plt.colorbar();

plt.subplot(144);
plt.imshow(response_gpu_shared_memory, cmap="gray");
plt.title("response_gpu_shared_memory");
plt.colorbar();
plt.show()

response_gpu_naive_cropped = \
    response_gpu_naive[ template_half_height:num_pixels_y-template_half_height, template_half_width:num_pixels_x-template_half_width ]

diff     = numpy.abs(response_opencv - response_gpu_naive_cropped)
max_diff = numpy.ones_like(diff, dtype=numpy.float32)



print \
    "Difference between OpenCV and GPU naive results as a percentage of the maximum possible difference (should be less than 1%%): %0.2f%%" % \
    (100 * (numpy.linalg.norm(diff) / numpy.linalg.norm(max_diff)))
    
print

rcParams['figure.figsize'] = 14,4
#figsize(14,4)

plt.subplot(131);
plt.imshow(response_gpu_naive_cropped, cmap="gray");
plt.title("response_gpu_naive_cropped");
plt.colorbar();

plt.subplot(132);
plt.imshow(response_opencv, cmap="gray");
plt.title("response_opencv");
plt.colorbar();

plt.subplot(133);
plt.imshow(diff, cmap="gray");
plt.title("diff");
plt.colorbar();
plt.show()


diff     = numpy.abs(response_gpu_shared_memory - response_gpu_naive)
max_diff = numpy.ones_like(diff, dtype=numpy.float32)



print \
    "Difference between GPU naive and GPU shared memory results as a percentage of the maximum possible difference (should be 0%%): %0.2f%%" % \
    (100 * (numpy.linalg.norm(diff) / numpy.linalg.norm(max_diff)))
    
print

rcParams['figure.figsize'] = 14,4
#figsize(14,4)

plt.subplot(131);
plt.imshow(response_gpu_shared_memory, cmap="gray");
plt.title("response_gpu_shared_memory");
plt.colorbar();

plt.subplot(132);
plt.imshow(response_gpu_naive, cmap="gray");
plt.title("response_gpu_naive");
plt.colorbar();

plt.subplot(133);
plt.imshow(diff, cmap="gray");
plt.title("diff");
plt.colorbar();
plt.show()


source_module = pycuda.compiler.SourceModule \
(
"""
__global__ void compute_coordinates(
    ushort2* d_packed_coordinates,
    int      num_pixels_y,
    int      num_pixels_x
)
{
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        d_packed_coordinates[ image_index_1d ] = make_ushort2( image_index_2d.x, image_index_2d.y );
    }
}

__global__ void add_constant(
    float* d_input,
    float* d_output,
    float  constant,
    int    num_pixels_y,
    int    num_pixels_x
)
{
    int  ny             = num_pixels_y;
    int  nx             = num_pixels_x;
    int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
    int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

    if ( image_index_2d.x < nx && image_index_2d.y < ny )
    {
        d_output[ image_index_1d ] = d_input[ image_index_1d ] + constant;
    }
}

__global__ void remove_redness_from_coordinates(
    ushort2*       d_coordinates,
    unsigned char* d_r,
    unsigned char* d_g,
    unsigned char* d_b,
    unsigned char* d_r_output,
    int    num_coordinates,
    int    num_pixels_y,
    int    num_pixels_x,
    int    template_half_height,
    int    template_half_width
)
{
    int ny              = num_pixels_y;
    int nx              = num_pixels_x;
    int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    if ( global_index_1d < num_coordinates )
    {
        ushort2 image_index_2d = d_coordinates[ global_index_1d ];

        for ( int y = image_index_2d.y - template_half_height; y < image_index_2d.y + template_half_height; y++ )
        {
            for ( int x = image_index_2d.x - template_half_width; x < image_index_2d.x + template_half_width; x++ )
            {
                int2 image_offset_index_2d         = make_int2( x, y );
                int2 image_offset_index_2d_clamped =
                    make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
                int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;
                
                unsigned char g_value = d_g[ image_offset_index_1d_clamped ];
                unsigned char b_value = d_b[ image_offset_index_1d_clamped ];

                unsigned int gb_average = ( g_value + b_value ) / 2;

                d_r_output[ image_offset_index_1d_clamped ] = (unsigned char)gb_average;
            }
        }   
    }
}
"""
)

pycuda.driver.memcpy_dtod(r_output_device, r_device, int(num_pixels_y * num_pixels_x))

num_coordinates_to_recolor   = numpy.int32(40)
recolor_template_half_height = numpy.int32(7)
recolor_template_half_width  = numpy.int32(7)



compute_coordinates_function             = source_module.get_function("compute_coordinates")
add_constant_function                    = source_module.get_function("add_constant")
remove_redness_from_coordinates_function = source_module.get_function("remove_redness_from_coordinates")


compute_coordinates_function_block = (16,16,1)
num_blocks_y                       = int(ceil(float(num_pixels_y) / float(compute_coordinates_function_block[1])))
num_blocks_x                       = int(ceil(float(num_pixels_x) / float(compute_coordinates_function_block[0])))
compute_coordinates_function_grid  = (num_blocks_x, num_blocks_y)

add_constant_function_block = (16,16,1)
num_blocks_y                = int(ceil(float(num_pixels_y) / float(add_constant_function_block[1])))
num_blocks_x                = int(ceil(float(num_pixels_x) / float(add_constant_function_block[0])))
add_constant_function_grid  = (num_blocks_x, num_blocks_y)

remove_redness_from_coordinates_function_block = (512,1,1)
num_blocks                                     = int(ceil(float(num_coordinates_to_recolor) / float(remove_redness_from_coordinates_function_block[0])))
remove_redness_from_coordinates_function_grid  = (num_blocks, 1)

compute_coordinates_function(
    coordinates_device,
    num_pixels_y,
    num_pixels_x,
    block=compute_coordinates_function_block,
    grid=compute_coordinates_function_grid)

pycuda.driver.memcpy_dtoh(coordinates_gpu, coordinates_device)


print coordinates_gpu
print

rcParams['figure.figsize'] = 4,4
#figsize(4,4)

plt.imshow(coordinates_gpu, cmap="gray");
plt.title("coordinates_gpu");
plt.colorbar();
plt.show()


add_constant_function(
    response_device,
    positive_response_device,
    numpy.float32(1.0),
    num_pixels_y,
    num_pixels_x,
    block=compute_coordinates_function_block,
    grid=compute_coordinates_function_grid)

pycuda.driver.memcpy_dtoh(positive_response_gpu, positive_response_device)


rcParams['figure.figsize'] = 4,4
#figsize(4,4)

plt.imshow(positive_response_gpu, cmap="gray");
plt.title("positive_response_gpu");
plt.colorbar();
plt.show()


radix_sort_manager.radix_sort_key_value_descending_device(
    positive_response_device,
    coordinates_device,
    sorted_positive_response_device,
    sorted_coordinates_device,
    int(num_pixels_y * num_pixels_x))

pycuda.driver.memcpy_dtoh(sorted_positive_response_gpu, sorted_positive_response_device)
pycuda.driver.memcpy_dtoh(sorted_coordinates_gpu, sorted_coordinates_device)


rcParams['figure.figsize'] = 9,4
#figsize(9,4)

plt.subplot(121);
plt.plot(sorted_positive_response_gpu.ravel());
plt.title("sorted_positive_response_gpu");

plt.subplot(122);
plt.imshow(sorted_coordinates_gpu, cmap="gray");
plt.title("sorted_coordinates_gpu");
plt.colorbar();
plt.show()


maximum_responses_image = zeros_like(sorted_positive_response_gpu)
sorted_coordinates_view = sorted_coordinates_gpu.ravel().view(dtype="u2, u2")

for i in range(40):
    coordinate = sorted_coordinates_view[i]
    maximum_responses_image[coordinate[1],coordinate[0]] = 1

rcParams['figure.figsize'] = 9,4
#figsize(9,4)

plt.subplot(121)
plt.imshow(maximum_responses_image, cmap="gray");
plt.title("sorted_positive_response_gpu");
plt.colorbar();

plt.subplot(122)
plt.imshow(response_opencv > 0.15, cmap="gray");
plt.title("response_opencv > 0.15");
plt.colorbar();
plt.show()


pycuda.driver.memcpy_dtod(r_output_device, r_device, int(num_pixels_y * num_pixels_x))

num_coordinates_to_recolor   = numpy.int32(40)
recolor_template_half_height = numpy.int32(7)
recolor_template_half_width  = numpy.int32(7)

remove_redness_from_coordinates_function(
    sorted_coordinates_device,
    r_device,
    g_device,
    b_device,
    r_output_device,
    num_coordinates_to_recolor,
    num_pixels_y,
    num_pixels_x,
    recolor_template_half_width,
    recolor_template_half_height,
    block=remove_redness_from_coordinates_function_block,
    grid=remove_redness_from_coordinates_function_grid)

pycuda.driver.memcpy_dtoh(r_output_gpu, r_output_device)

rgba_output_gpu = numpy.concatenate((r_output_gpu[:,:,newaxis],g[:,:,newaxis],b[:,:,newaxis],a[:,:,newaxis]), axis=2).copy()


rcParams['figure.figsize'] = 9,4
#figsize(9,4)

plt.subplot(121);
plt.imshow(rgba);
plt.title("rgba");

plt.subplot(122);
plt.imshow(rgba_output_gpu);
plt.title("rgba_output_gpu");
plt.show()
