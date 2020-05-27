#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>

#include "numpy/arrayobject.h"
#include "momentum/momentum_backend.h"
#include "error_methods.h"
#include "generic_simd.h"

static PyObject* RSI(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in;
    int window_size = 14;

    static char *kwlist[] = {
        "close",
        "n",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:RSI", kwlist,
                          &in,
                          &window_size)) {
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
            double* rsi = _RSI_DOUBLE(close, close_len, window_size);
            npy_intp dims[1] = {close_len};

            Py_DECREF(arr);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, rsi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(arr);
            float* rsi = _RSI_FLOAT(close, close_len, window_size);
            npy_intp dims[1] = {close_len};

            Py_DECREF(arr);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, rsi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* AO(PyObject* self, PyObject* args, PyObject* kwargs) {
    int n1 = 5;
    int n2 = 34;
    PyObject* in1;
    PyObject* in2;

    static char *kwlist[] = {
            "high",
            "low",
            "s",
            "l",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ii:AO", kwlist,
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

            if (!check_double_align(high) ||
                !check_double_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            double* ao = _AO_DOUBLE(high, low, n1, n2, high_len);
            npy_intp dims[1] = {high_len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, ao);
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

            float* ao = _AO_FLOAT(high, low, n1, n2, high_len);
            npy_intp dims[1] = {high_len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, ao);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* KAMA(PyObject* self, PyObject* args, PyObject* kwargs) {
    int n1 = 10;
    int n2 = 2;
    int n3 = 30;
    PyObject* in1;

    static char *kwlist[] = {
            "close",
            "n",
            "f",
            "s",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iii:KAMA", kwlist,
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

            if (!check_double_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            double* kama = _KAMA_DOUBLE(close, n1, n2, n3, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, kama);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);

            if (!check_float_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            float* kama = _KAMA_FLOAT(close, n1, n2, n3, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, kama);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* ROC(PyObject* self, PyObject* args, PyObject* kwargs) {
    int n = 12;
    PyObject* in;

    static char *kwlist[] = {
            "close",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i:ROC", kwlist,
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

            if (!check_double_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            double* roc = _ROC_DOUBLE(close, n, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, roc);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);

            if (!check_float_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            float* roc = _ROC_FLOAT(close, n, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, roc);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* STOCHASTIC_OSCILLATOR(PyObject* self, PyObject* args, PyObject* kwargs) {
    int n = 14;
    int d = 3;
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "n",
            "d_n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ii:StochasticOscillator", kwlist,
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

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            double* so = _STOCHASTIC_OSCILLATOR_DOUBLE(high,
                    low, close, n, d, len);
            npy_intp dims[2] = {2, len};

            Py_DECREF(high);
            Py_DECREF(low);
            Py_DECREF(close);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, so);
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

            float* so = _STOCHASTIC_OSCILLATOR_FLOAT(high,
                                                     low, close, n, d, len);
            npy_intp dims[2] = {2, len};

            Py_DECREF(high);
            Py_DECREF(low);
            Py_DECREF(close);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, so);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* TSI(PyObject* self, PyObject* args, PyObject* kwargs) {
    int r = 25;
    int s = 13;
    PyObject* in;

    static char *kwlist[] = {
            "close",
            "r",
            "s",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii:TSI", kwlist,
                          &in,
                          &r,
                          &s)) {
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

            if (!check_double_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            double* tsi = _TSI_DOUBLE(close, r, s, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, tsi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);

            if (!check_float_align(close)) {
                raise_alignment_error();
                return NULL;
            }

            float* tsi = _TSI_FLOAT(close, r, s, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, tsi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* ULTIMATE_OSCILLATOR(PyObject* self, PyObject* args, PyObject* kwargs) {
    int s = 7;
    int m = 14;
    int l = 28;
    double ws = 4.f;
    double wm = 2.f;
    double wl = 1.f;
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "s",
            "m",
            "l",
            "ws",
            "wm",
            "wl",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|iiiddd:UltimateOscillator", kwlist,
                          &in1,
                          &in2,
                          &in3,
                          &s,
                          &m,
                          &l,
                          &ws,
                          &wm,
                          &wl)) {
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
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) || len != PyArray_SIZE(_close)) {
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

            double* uo = _ULTIMATE_OSCILLATOR_DOUBLE(high, low, close, s, m, l, ws, wm, wl, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, uo);
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

            float* uo = _ULTIMATE_OSCILLATOR_FLOAT(high, low, close, s, m, l, ws, wm, wl, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, uo);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* WILLIAMS_R(PyObject* self, PyObject* args, PyObject* kwargs) {
    int n = 14;
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i:WilliamsR", kwlist,
                          &in1,
                          &in2,
                          &in3,
                          &n)) {
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

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low)) {
                raise_alignment_error();
                return NULL;
            }

            double* wr = _WILLIAMS_R_DOUBLE(high, low, close, n, len);
            npy_intp dims[1] = {len};

            Py_DECREF(high);
            Py_DECREF(low);
            Py_DECREF(close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, wr);
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

            float* wr = _WILLIAMS_R_FLOAT(high, low, close, n, len);
            npy_intp dims[1] = {len};

            Py_DECREF(high);
            Py_DECREF(low);
            Py_DECREF(close);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, wr);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
            return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

#define pyargflag METH_VARARGS | METH_KEYWORDS

static PyMethodDef MomentumMethods[] = {
        {"RSI", (PyCFunction) RSI, pyargflag, "Compute RSI On Data"},
        {"AO", (PyCFunction) AO, pyargflag, "Compute AO On Data"},
        {"KAMA", (PyCFunction) KAMA, pyargflag, "Compute KAMA On Data"},
        {"ROC", (PyCFunction) ROC, pyargflag, "Compute ROC On Data"},
        {"StochasticOscillator", (PyCFunction) STOCHASTIC_OSCILLATOR, pyargflag, "Compute Stochastic Oscillator On Data"},
        {"TSI", (PyCFunction) TSI, pyargflag, "Compute TSI On Data"},
        {"UltimateOscillator", (PyCFunction) ULTIMATE_OSCILLATOR, pyargflag, "Compute Ultimate Oscillator On Data"},
        {"WilliamsR", (PyCFunction) WILLIAMS_R, pyargflag, "Compute Williams %R On Data"},
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
