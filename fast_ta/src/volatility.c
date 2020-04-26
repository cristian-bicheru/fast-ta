#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>

#include "2darray.h"
#include "numpy/arrayobject.h"
#include "volatility_backend.h"
#include "error_methods.h"

static PyObject* ATR(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    int n = 14;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i:ATR", kwlist,
                                     &in1,
                                     &in2,
                                     &in3,
                                     &n)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2) ||
        type1 != PyArray_TYPE((PyArrayObject*) in3)) {
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
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) ||
        len != PyArray_SIZE(_close)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* close = PyArray_DATA(_close);
            double* atr = _ATR_DOUBLE(high, low, close, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), atr,
                   len*sizeof(double));
            free(atr);
            return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            float* atr = _ATR_FLOAT(high, low, close, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), atr,
                   len*sizeof(float));
            free(atr);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* BOL(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    int n = 20;
    float ndev = 2;

    static char *kwlist[] = {
            "close",
            "n",
            "ndev",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|if:BOL", kwlist,
                                     &in1,
                                     &n,
                                     &ndev)) {
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
            double** bol = _BOL_DOUBLE(close, len, n, ndev);
            npy_intp dims[1] = {len};

            PyObject* ret = PyTuple_New(3);
            PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) arr1), bol[0],
                   len*sizeof(double));
            PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) arr2), bol[1],
                   len*sizeof(double));
            PyObject* arr3 = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) arr3), bol[2],
                   len*sizeof(double));
            double_free2d(bol, 3);
            PyTuple_SetItem(ret, 0, arr1);
            PyTuple_SetItem(ret, 1, arr2);
            PyTuple_SetItem(ret, 2, arr3);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float** bol = _BOL_FLOAT(close, len, n, ndev);
            npy_intp dims[1] = {len};

            PyObject* ret = PyTuple_New(3);
            PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) arr1), bol[0],
                   len*sizeof(float));
            PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) arr2), bol[1],
                   len*sizeof(float));
            PyObject* arr3 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) arr3), bol[2],
                   len*sizeof(float));
            float_free2d(bol, 3);
            PyTuple_SetItem(ret, 0, arr1);
            PyTuple_SetItem(ret, 1, arr2);
            PyTuple_SetItem(ret, 2, arr3);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

#define pyargflag METH_VARARGS | METH_KEYWORDS

static PyMethodDef VolumeMethods[] = {
        {"ATR", (PyCFunction) ATR, pyargflag, "Compute ATR On Data"},
        {"BOL", (PyCFunction) BOL, pyargflag, "Compute Bollinger Bands On Data"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "volatility", "doc todo",
        -1,
        VolumeMethods
};

PyMODINIT_FUNC
PyInit_volatility(void) {
    import_array();
    return PyModule_Create(&cModPyDem);
};
