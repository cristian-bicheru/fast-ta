#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>

#include "numpy/arrayobject.h"
#include "volume_backend.h"
#include "error_methods.h"

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
            double* adi = _ADI_DOUBLE(high, low, close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            memcpy(PyArray_DATA((PyArrayObject*) ret), adi,
                   len*sizeof(double));
            free(adi);
            return ret;
        }
        case NPY_FLOAT32: {
            float* high = PyArray_DATA(_high);
            float* low = PyArray_DATA(_low);
            float* close = PyArray_DATA(_close);
            float* volume = PyArray_DATA(_volume);
            float* adi = _ADI_FLOAT(high, low, close, volume, len);
            npy_intp dims[1] = {len};

            Py_DECREF(_high);
            Py_DECREF(_low);
            PyObject* ret = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            memcpy(PyArray_DATA((PyArrayObject*) ret), adi,
                   len*sizeof(float));
            free(adi);
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
