#include "Python.h" 

static PyObject * 

hello_opencv(PyObject *self, PyObject *args) 

{ 
    return Py_BuildValue("s", "Hello OpenCV!"); 
    } 
// creation of a functions table 
static PyMethodDef 
module_functions[] = { 
    { "hello_opencv", hello_opencv, METH_VARARGS, "Say Hello" }, 
    { NULL } 
}; 
void 

inithello_opencv(void) 

{ 

    Py_InitModule3("hello_opencv", module_functions, "Hello Python C API"); 

}
