#include "Python.h"

#include <stdlib.h>

#include "error_methods.h"

void raise_dtype_error() {
    PyErr_SetString(PyExc_TypeError, "Unsupported DType");
}

void raise_error(char* msg) {
    PyErr_SetString(PyExc_TypeError, msg);
}

void raise_alignment_error() {
    PyErr_SetString(PyExc_TypeError, "Input Array Unaligned");
}