#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>

#include "2darray.h"
#include "numpy/arrayobject.h"
#include "volatility/volatility_backend.h"
#include "error_methods.h"
#include "generic_simd/generic_simd.h"

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

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            double* atr = _ATR_DOUBLE(high, low, close, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, atr);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);

            if (!check_float_align(close) ||
                !check_float_align(high) ||
                !check_float_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            float* atr = _ATR_FLOAT(high, low, close, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, atr);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
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

            if (!check_double_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            double** bol = _BOL_DOUBLE(close, len, n, ndev);
            npy_intp dims[2] = {3, len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, bol[0]);
            free(bol);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);

            if (!check_float_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            float** bol = _BOL_FLOAT(close, len, n, ndev);
            npy_intp dims[2] = {3, len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, bol[0]);
            free(bol);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* DC(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    int n = 20;

    static char *kwlist[] = {
            "high",
            "low",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i:DC", kwlist,
                                     &in1,
                                     &in2,
                                     &n)) {
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
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);

            if (!check_double_align(high) ||
                !check_double_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            double** dc = _DC_DOUBLE(high, low, len, n);
            npy_intp dims[2] = {3, len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, dc[0]);
            free(dc);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);

            if (!check_float_align(high) ||
                !check_float_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            float** dc = _DC_FLOAT(high, low, len, n);
            npy_intp dims[2] = {3, len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, dc[0]);
            free(dc);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* KC(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    int n1 = 14;
    int n2 = 10;
    int num_channels = 1;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "n1",
            "n2",
            "num_channels",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|iii:KC", kwlist,
                                     &in1,
                                     &in2,
                                     &in3,
                                     &n1,
                                     &n2,
                                     &num_channels)) {
        return NULL;
    }

    if (num_channels < 0) {
        raise_error("Number of extra channels cannot be negative.");
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

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            double** kc = _KC_DOUBLE(high, low, close, len, n1, n2, num_channels);
            npy_intp dims[2] = {2*num_channels+1, len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, kc[0]);
            free(kc);
			PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);

            if (!check_float_align(close) ||
                !check_float_align(high) ||
                !check_float_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            float** kc = _KC_FLOAT(high, low, close, len, n1, n2, num_channels);
            npy_intp dims[2] = {2*num_channels+1, len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, kc[0]);
            free(kc);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
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
        {"DC", (PyCFunction) DC, pyargflag, "Compute Donchian Channels On Data"},
        {"KC", (PyCFunction) KC, pyargflag, "Compute Keltner Channels On Data"},
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
