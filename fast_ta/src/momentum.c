#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>

#include "numpy/arrayobject.h"
#include "array_pair.h"
#include "momentum_backend.h"
#include "parallel_momentum_backend.h"
#include "error_methods.h"

static PyObject* RSI(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in;
    int window_size;
    int thread_count = 1;

    static char *kwlist[] = {
        "",
        "",
        "threads",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i:RSI", kwlist,
                          &in,
                          &window_size,
                          &thread_count)) {
        printf("Exception\n");
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
            double* rsi = _PARALLEL_RSI_DOUBLE(close, close_len, window_size,
                                               thread_count);
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
            float* rsi = _PARALLEL_RSI_FLOAT(close, close_len, window_size,
                                             thread_count);
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
        raise_error("Input Array DType Mismatch");
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
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), kama,
                   len*sizeof(double));
            free(kama);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* kama = _KAMA_FLOAT(close, n1, n2, n3, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), kama,
                   len*sizeof(float));
            free(kama);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* ROC(PyObject* self, PyObject* args) {
    int n;
    PyObject* in;

    if (!PyArg_ParseTuple(args, "Oi",
                          &in,
                          &n)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in);

    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in,
                                                              type1,
                                                              NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    switch(type1) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(_close);
            double* roc = _ROC_DOUBLE(close, n, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), roc,
                   len*sizeof(double));
            free(roc);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* roc = _ROC_FLOAT(close, n, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), roc,
                   len*sizeof(float));
            free(roc);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* STOCHASTIC_OSCILLATOR(PyObject* self, PyObject* args) {
    int n;
    int d;
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;

    if (!PyArg_ParseTuple(args, "OOOii",
                          &in1,
                          &in2,
                          &in3,
                          &n,
                          &d)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2) || type1 != PyArray_TYPE((PyArrayObject*) in3)) {
        raise_error("Input Array DType Mismatch");
        return NULL;
    }

    PyArrayObject* _high = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                             type1,
                                                             NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _low = (PyArrayObject*) PyArray_FROM_OTF(in2,
                                                            type1,
                                                            NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in3,
                                                              type1,
                                                              NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    if (len != PyArray_SIZE(_high) || len != PyArray_SIZE(_low)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* close = PyArray_DATA(_close);
            struct double_array_pair so = _STOCHASTIC_OSCILLATOR_DOUBLE(high, low, close, n, d, len);
            npy_intp dims[1] = {len};

            PyObject* ret = PyTuple_New(2);
            PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) arr1), so.arr1,
                   len*sizeof(double));
            PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) arr2), so.arr2,
                   len*sizeof(double));
            free(so.arr1);
            free(so.arr2);
            PyTuple_SetItem(ret, 0, arr1);
            PyTuple_SetItem(ret, 1, arr2);
            return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            struct float_array_pair so = _STOCHASTIC_OSCILLATOR_FLOAT(high, low, close, n, d, len);
            npy_intp dims[1] = {len};

            PyObject* ret = PyTuple_New(2);
            PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) arr1), so.arr1,
                   len*sizeof(float));
            PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) arr2), so.arr2,
                   len*sizeof(float));
            free(so.arr1);
            free(so.arr2);
            PyTuple_SetItem(ret, 0, arr1);
            PyTuple_SetItem(ret, 1, arr2);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyMethodDef MomentumMethods[] = {
        {"RSI", RSI, METH_VARARGS | METH_KEYWORDS, "Compute RSI On Data"},
        {"AO", AO, METH_VARARGS, "Compute AO On Data"},
        {"KAMA", KAMA, METH_VARARGS, "Compute KAMA On Data"},
        {"ROC", ROC, METH_VARARGS, "Compute ROC On Data"},
        {"StochasticOscillator", STOCHASTIC_OSCILLATOR, METH_VARARGS, "Compute Stochastic Oscillator On Data"},
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
