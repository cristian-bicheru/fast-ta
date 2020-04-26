#include <stdlib.h>

double** double_malloc2d(int n, int len) {
    double** arr = malloc(n*sizeof(double*));
    for (int i = 0; i < n; i++) {
        arr[i] = malloc(len*sizeof(double));
    }
    return arr;
}

void double_free2d(double** narr, int n) {
    for (int i = 0; i < n; i++) {
        free(narr[i]);
    }
    free(narr);
}

float** float_malloc2d(int n, int len) {
    float** arr = malloc(n*sizeof(float*));
    for (int i = 0; i < n; i++) {
        arr[i] = malloc(len*sizeof(float));
    }
    return arr;
}

void float_free2d(float** narr, int n) {
    for (int i = 0; i < n; i++) {
        free(narr[i]);
    }
    free(narr);
}