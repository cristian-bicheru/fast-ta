#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <stdio.h>
#include "momentum_backend.c"
#include <stdlib.h>
#include "error_methods.c"

static PyObject* RSI(PyObject* self, PyObject* args) {
    int _n;
    PyObject* in;

    if (!PyArg_ParseTuple(args, "Oi",
                          &in,
                          &_n)) {
        return NULL;
    }

    int type = PyArray_TYPE((PyArrayObject*) in);
    PyArrayObject* arr = (PyArrayObject*) PyArray_FROM_OTF(in,
                                                           type,
                                                           NPY_ARRAY_IN_ARRAY);
    int close_len = PyArray_SIZE(arr);

    switch(type) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(arr);
            double* rsi = _RSI_DOUBLE(close, close_len, _n);
            npy_intp dims[1] = {close_len};

            Py_DECREF(arr);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), rsi,
                             close_len*sizeof(double));
            free(rsi);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(arr);
            float* rsi = _RSI_FLOAT(close, close_len, _n);
            npy_intp dims[1] = {close_len};

            Py_DECREF(arr);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), rsi,
                             close_len*sizeof(float));
            free(rsi);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* AO(PyObject* self, PyObject* args) {
    int n1;
    int n2;
    PyObject* in1;
    PyObject* in2;

    if (!PyArg_ParseTuple(args, "OOii",
                          &in1,
                          &in2,
                          &n1,
                          &n2)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2)) {
        raise_error("Input Arrays Have Different DTypes");
        return NULL;
    }

    PyArrayObject* _high = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                             type1,
                                                             NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _low = (PyArrayObject*) PyArray_FROM_OTF(in2,
                                                            type1,
                                                            NPY_ARRAY_IN_ARRAY);
    int high_len = PyArray_SIZE(_high);

    if (high_len != PyArray_SIZE(_low)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* ao = _AO_DOUBLE(high, low, n1, n2, high_len);
            npy_intp dims[1] = {high_len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), ao,
                             high_len*sizeof(double));
            free(ao);
            return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* ao = _AO_FLOAT(high, low, n1, n2, high_len);
            npy_intp dims[1] = {high_len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), ao,
                             high_len*sizeof(float));
            free(ao);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* KAMA(PyObject* self, PyObject* args) {
    int n1;
    int n2;
    int n3;
    PyObject* in1;

    if (!PyArg_ParseTuple(args, "Oiii",
                          &in1,
                          &n1,
                          &n2,
                          &n3)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                              type1,
                                                              NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    switch(type1) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(_close);
            double* kama = _KAMA_DOUBLE(close, n1, n2, n3, len);
            npy_intp dims[1] = {len-n1};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), kama,
                   (len-n1)*sizeof(double));
            free(kama);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* kama = _KAMA_FLOAT(close, n1, n2, n3, len);
            npy_intp dims[1] = {len-n1};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), kama,
                   (len-n1)*sizeof(float));
            free(kama);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyMethodDef MomentumMethods[] = {
        {"RSI", RSI, METH_VARARGS, "Compute RSI On Data"},
        {"AO", AO, METH_VARARGS, "Compute AO On Data"},
        {"KAMA", KAMA, METH_VARARGS, "Compute KAMA On Data"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "momentum", "doc todo",
        -1,
        MomentumMethods
};

PyMODINIT_FUNC
PyInit_momentum(void) {
    import_array();
    return PyModule_Create(&cModPyDem);
};
