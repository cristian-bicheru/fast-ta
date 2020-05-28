#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>

#include "numpy/arrayobject.h"
#include "volume/volume_backend.h"
#include "error_methods.h"
#include "generic_simd/generic_simd.h"

static PyObject* ADI(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    PyObject* in4;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "volume",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO:ADI", kwlist,
                                     &in1,
                                     &in2,
                                     &in3,
                                     &in4)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2) ||
        type1 != PyArray_TYPE((PyArrayObject*) in3) ||
        type1 != PyArray_TYPE((PyArrayObject*) in4)) {
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
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in4,
                                                            type1,
                                                            NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) ||
        len != PyArray_SIZE(_close) ||
        len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* adi = _ADI_DOUBLE(high, low, close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, adi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(high) ||
                !check_float_align(low) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* adi = _ADI_FLOAT(high, low, close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, adi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* CMF(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    PyObject* in4;
    int n = 20;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "volume",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|i:CMF", kwlist,
                                     &in1,
                                     &in2,
                                     &in3,
                                     &in4,
                                     &n)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2) ||
        type1 != PyArray_TYPE((PyArrayObject*) in3) ||
        type1 != PyArray_TYPE((PyArrayObject*) in4)) {
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
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in4,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) ||
        len != PyArray_SIZE(_close) ||
        len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* cmf = _CMF_DOUBLE(high, low, close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, cmf);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(high) ||
                !check_float_align(low) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* cmf = _CMF_FLOAT(high, low, close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, cmf);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* EMV(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    int n = 14;

    static char *kwlist[] = {
            "high",
            "low",
            "volume",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i:EMV", kwlist,
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
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in3,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) ||
        len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(high) ||
                !check_double_align(low) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double** emv = _EMV_DOUBLE(high, low, volume, len, n);
            npy_intp dims[2] = {2, len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, emv[0]);
            free(emv);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(high) ||
                !check_float_align(low) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float** emv = _EMV_FLOAT(high, low, volume, len, n);
            npy_intp dims[2] = {2, len};
            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, emv[0]);
            free(emv);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* FI(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    int n = 13;

    static char *kwlist[] = {
            "close",
            "volume",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i:FI", kwlist,
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

    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                             type1,
                                                             NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in2,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    if (len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* fi = _FI_DOUBLE(close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, fi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* fi = _FI_FLOAT(close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* MFI(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    PyObject* in4;
    int n = 14;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "volume",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|i:MFI", kwlist,
                                     &in1,
                                     &in2,
                                     &in3,
                                     &in4,
                                     &n)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2) ||
        type1 != PyArray_TYPE((PyArrayObject*) in3) ||
        type1 != PyArray_TYPE((PyArrayObject*) in4)) {
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
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in4,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) ||
        len != PyArray_SIZE(_close) ||
        len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* mfi = _MFI_DOUBLE(high, low, close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, mfi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(high) ||
                !check_float_align(low) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* mfi = _MFI_FLOAT(high, low, close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, mfi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* NVI(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;

    static char *kwlist[] = {
            "close",
            "volume",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:NVI", kwlist,
                                     &in1,
                                     &in2)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2)) {
        raise_error("Input Array DType Mismatch");
        return NULL;
    }

    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                              type1,
                                                              NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in2,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    if (len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* nvi = _NVI_DOUBLE(close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, nvi);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* nvi = _NVI_FLOAT(close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, nvi);

            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* OBV(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;

    static char *kwlist[] = {
            "close",
            "volume",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:OBV", kwlist,
                                     &in1,
                                     &in2)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2)) {
        raise_error("Input Array DType Mismatch");
        return NULL;
    }

    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                              type1,
                                                              NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in2,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    if (len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* obv = _OBV_DOUBLE(close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, obv);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* obv = _OBV_FLOAT(close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, obv);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* VPT(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;

    static char *kwlist[] = {
            "close",
            "volume",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:VPT", kwlist,
                                     &in1,
                                     &in2)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2)) {
        raise_error("Input Array DType Mismatch");
        return NULL;
    }

    PyArrayObject* _close = (PyArrayObject*) PyArray_FROM_OTF(in1,
                                                              type1,
                                                              NPY_ARRAY_IN_ARRAY);
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in2,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_close);

    if (len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* vpt = _VPT_DOUBLE(close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, vpt);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* vpt = _VPT_FLOAT(close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, vpt);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        default:
            raise_dtype_error();
            return NULL;
    }
};

static PyObject* VWAP(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* in1;
    PyObject* in2;
    PyObject* in3;
    PyObject* in4;
    int n = 14;

    static char *kwlist[] = {
            "high",
            "low",
            "close",
            "volume",
            "n",
            NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|i:MFI", kwlist,
                                     &in1,
                                     &in2,
                                     &in3,
                                     &in4,
                                     &n)) {
        return NULL;
    }

    int type1 = PyArray_TYPE((PyArrayObject*) in1);

    if (type1 != PyArray_TYPE((PyArrayObject*) in2) ||
        type1 != PyArray_TYPE((PyArrayObject*) in3) ||
        type1 != PyArray_TYPE((PyArrayObject*) in4)) {
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
    PyArrayObject* _volume = (PyArrayObject*) PyArray_FROM_OTF(in4,
                                                               type1,
                                                               NPY_ARRAY_IN_ARRAY);
    int len = PyArray_SIZE(_high);

    if (len != PyArray_SIZE(_low) ||
        len != PyArray_SIZE(_close) ||
        len != PyArray_SIZE(_volume)) {
        raise_error("Input Array Dim Mismatch");
        return NULL;
    }

    switch(type1) {
        case NPY_FLOAT64: {
            double* high = PyArray_DATA(_high);
            double* low = PyArray_DATA(_low);
            double* close = PyArray_DATA(_close);
            double* volume = PyArray_DATA(_volume);

            if (!check_double_align(close) ||
                !check_double_align(high) ||
                !check_double_align(low) ||
                !check_double_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            double* vwap = _VWAP_DOUBLE(high, low, close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, vwap);
            PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA); 
			return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);

            if (!check_float_align(close) ||
                !check_float_align(high) ||
                !check_float_align(low) ||
                !check_float_align(volume)) {
                raise_alignment_error();
                return NULL;
            }

            float* vwap = _VWAP_FLOAT(high, low, close, volume, len, n);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            Py_DECREF(_close);
            Py_DECREF(_volume);
            PyObject* ret = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, vwap);
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
        {"ADI", (PyCFunction) ADI, pyargflag, "Compute ADI On Data"},
        {"CMF", (PyCFunction) CMF, pyargflag, "Compute CMF On Data"},
        {"EMV", (PyCFunction) EMV, pyargflag, "Compute EMV On Data"},
        {"FI", (PyCFunction) FI, pyargflag, "Compute FI On Data"},
        {"MFI", (PyCFunction) MFI, pyargflag, "Compute MFI On Data"},
        {"NVI", (PyCFunction) NVI, pyargflag, "Compute NVI On Data"},
        {"OBV", (PyCFunction) OBV, pyargflag, "Compute OBV On Data"},
        {"VPT", (PyCFunction) VPT, pyargflag, "Compute VPT On Data"},
        {"VWAP", (PyCFunction) VWAP, pyargflag, "Compute VWAP On Data"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "volume", "doc todo",
        -1,
        VolumeMethods
};

PyMODINIT_FUNC
PyInit_volume(void) {
    import_array();
    return PyModule_Create(&cModPyDem);
};
