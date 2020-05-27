#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "numpy/arrayobject.h"
#include "error_methods.h"
#include "core/core_backend.h"


static PyObject* align(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in;

    if (!PyArg_ParseTuple(args, "O:align", &in)) {
        return NULL;
    }

    int type = PyArray_TYPE((PyArrayObject*) in);
    PyArrayObject* arr = (PyArrayObject*) PyArray_FROM_OTF(in,
                                                           type,
                                                           NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(arr);

    switch(type) {
        case NPY_FLOAT64: {
            double* data = PyArray_DATA(arr);
            double* aligned = _ALIGN_DOUBLE(data, len);
            npy_intp* dims = PyArray_DIMS(arr);

            Py_DECREF(arr);
            PyObject* ret = PyArray_SimpleNewFromData(PyArray_NDIM(arr), dims, NPY_FLOAT64, aligned);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        case NPY_FLOAT32: {
            float* data = PyArray_DATA(arr);
            float* aligned = _ALIGN_FLOAT(data, len);
            npy_intp* dims = PyArray_DIMS(arr);

            Py_DECREF(arr);
            PyObject* ret = PyArray_SimpleNewFromData(PyArray_NDIM(arr), dims, NPY_FLOAT32, aligned);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

const char* isetnames[6] = {
        "SSE2",
        "SSE4.1",
        "AVX",
        "AVX2",
        "FMA",
        "AVX512F",
};

const char pydictstr[25] = "{s:i,s:i,s:i,s:i,s:i,s:i}";

bool flags[6] = {0};

static PyObject* iset(PyObject* self, PyObject* args, PyObject* kwargs) {
    #ifdef SSE2
        flags[0] = true;
    #endif
    #ifdef SSE41
        flags[1] = true;
    #endif
    #ifdef AVX
        flags[2] = true;
    #endif
    #ifdef AVX2
        flags[3] = true;
    #endif
    #ifdef FMA
        flags[4] = true;
    #endif
    #ifdef AVX512F
        flags[5] = true;
    #endif

    PyObject* support = Py_BuildValue(pydictstr, isetnames[0], flags[0],
                                      isetnames[1], flags[1],
                                      isetnames[2], flags[2],
                                      isetnames[3], flags[3],
                                      isetnames[4], flags[4],
                                      isetnames[5], flags[5]);
    return support;
};

#define pyargflag METH_VARARGS | METH_KEYWORDS

static PyMethodDef CoreMethods[] = {
        {"align", (PyCFunction) align, pyargflag, "Return an aligned copy of a NumPy array."},
        {"instruction_set", (PyCFunction) iset, pyargflag, "Return the SIMD Instruction Set fast-ta was compiled with."},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "core", "doc todo",
        -1,
        CoreMethods
};

PyMODINIT_FUNC
PyInit_core(void) {
    import_array();
    return PyModule_Create(&cModPyDem);
};
