from distutils.core import setup, Extension 

setup(
    name = "chapter1", 
    ext_modules = [ 
        Extension("hello_opencv", sources=["hello_opencv.c"]),
	],
	)
