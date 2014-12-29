from distutils.core import setup, Extension 

extension_mod = Extension("_swigversion", 
			["_swigversion_module.cpp", "hello.cpp"]) 
setup(name = "swigversion", ext_modules=[extension_mod])
