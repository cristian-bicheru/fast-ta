#pragma once

#include "Python.h"

#include <stdlib.h>
#include <math.h>

#include "debug_tools.h"

void print(char* str) {
    PyObject* pyBuiltIn = PyImport_ImportModule("builtins");
    PyObject* tmp = PyObject_CallMethod(pyBuiltIn, "print", "s", str);
    if (tmp != NULL) Py_DECREF(tmp);
    Py_DECREF(pyBuiltIn);
};

char* int_to_str(int i) {
    int len = floor(log10(i))+1;
    char* ret = malloc(len+1);

    ret[len] = '\0';

    for (int j = 0; j < len; j++) {
        ret[len-j-1] = '0'+i%10;
        i = floor(i/10.);
    }

    return ret;
};

void print_int(int i) {
    if (i != 0) {
        char* st;
        st = int_to_str(i);
        print(st);
        free(st);
    } else {
        print("0");
    }
}
