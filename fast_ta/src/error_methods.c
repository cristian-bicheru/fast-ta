#include "Python.h"
#include <stdlib.h>

const char* dtype_error_message = "Unsupported DType";

void raise_dtype_error() {
    PyErr_SetString(PyExc_TypeError, dtype_error_message);
}

void raise_error(char* msg) {
    PyErr_SetString(PyExc_TypeError, msg);
}
