#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

# cuda code between the quotes that makes the addition possible
mod = SourceModule("""
__global__ void add_five(float *dest, float *a)
{
  const int i = threadIdx.x;
  dest[i] = 5.+ a[i];
}
""")

# get the function add_five
add_five = mod.get_function("add_five")

# random production of 50 random numbers
a = np.random.randn(50).astype(np.float32)

# destination for the answer in the same dimensions of the random-produced saved in a
dest = np.zeros_like(a)
add_five(
        cuda.Out(dest), cuda.In(a), 
        block=(50,1,1), grid=(1,1))

# print the random-generated numbers
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print a

# print the random-generated numbers + the added value (5)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print dest
