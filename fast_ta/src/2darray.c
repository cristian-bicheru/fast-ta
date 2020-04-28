#include <stdlib.h>

/**
  * NOTE: Numpy Expects Arrays To Be Memory-contiguous
  */

double** double_malloc2d(int n, int len) {
    double* arr = malloc(n*len*sizeof(double));
    double** ptrs = malloc(n*sizeof(double*));
    ptrs[0] = arr;

    for (int i = 1; i < n; i++) {
        ptrs[i] = ptrs[i-1]+len;
    }

    return ptrs;
}

void double_free2d(double** ptrarr) {
    free(ptrarr[0]);
    free(ptrarr);
}

float** float_malloc2d(int n, int len) {
    float* arr = malloc(n*len*sizeof(float));
    float** ptrs = malloc(n*sizeof(float*));
    ptrs[0] = arr;

    for (int i = 1; i < n; i++) {
        ptrs[i] = ptrs[i-1]+len;
    }

    return ptrs;
}

void float_free2d(float** ptrarr) {
    free(ptrarr[0]);
    free(ptrarr);
}