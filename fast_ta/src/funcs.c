#include "Python.h"
#include "numpy/arrayobject.h"
#include <immintrin.h>
#include <stdlib.h>

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

/**
 * Inplace Exponential Moving Average
 * @param arr
 * @param len
 * @param alpha
 */
void inplace_ema(double* arr, int len, double alpha) {
    for (int i = 1; i < len; i++) {
        arr[i] = arr[i-1] * (1-alpha) + arr[i] * alpha;
    }
}

/**
 * Vectorized Pairwise Mean Between Two Arrays
 * @param arr1
 * @param arr2
 * @param len
 * @return
 */
double* _double_pairwise_mean(double* arr1, double* arr2, int len) {
    double* median = aligned_alloc(256, len * sizeof(double));
    __m256d v1, v2;
    __m256d d2 = _mm256_set_pd(0.5, 0.5, 0.5, 0.5);

    for (int i = 0; i < len-len%4; i += 4) {
        v1 = _mm256_loadu_pd(&arr1[i]);
        v2 = _mm256_loadu_pd(&arr2[i]);
        v1 = _mm256_add_pd(v1, v2);
        v1 = _mm256_mul_pd(v1, d2);
        _mm256_stream_pd(&median[i], v1);
    }

    for (int i = len-len%4; i < len; i++) {
        median[i] = (arr1[i]+arr2[i])/2;
    }

    return median;
}

float* _float_pairwise_mean(float* arr1, float* arr2, int len) {
    float* median = aligned_alloc(256, len*sizeof(float));
    __m256 v1, v2;
    __m256 d2 = _mm256_set_ps(0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5);

    for (int i = 0; i < len-len%8; i += 8) {
        v1 = _mm256_loadu_ps(&arr1[i]);
        v2 = _mm256_loadu_ps(&arr2[i]);
        v1 = _mm256_add_ps(v1, v2);
        v1 = _mm256_mul_ps(v1, d2);
        _mm256_stream_ps(&median[i], v1);
    }

    for (int i = len-len%8; i < len; i++) {
        median[i] = (arr1[i]+arr2[i])/2;
    }

    return median;
}

/**
 * Vectorized Inplace Division
 * @param arr
 * @param len
 * @param x
 */
void double_inplace_div(double* arr, int len, double x) {
    __m256d v, vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&arr[i], _mm256_div_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        arr[i] = arr[i]/x;
    }
}

void float_inplace_div(float* arr, int len, float x) {
    __m256 v, vx;
    vx = _mm256_set_ps(x, x, x, x, x, x, x, x);

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&arr[i], _mm256_div_ps(v, vx));
    }

    for (int i = len-len%8; i < len; i++) {
        arr[i] = arr[i]/x;
    }
}

/**
 * Compute the Simple Moving Average on an array.
 * @param arr
 * @param len
 * @param window
 * @return sma
 */
double* _double_sma(const double* arr, int len, int window) {
    double wsum = 0;
    double* sma = malloc(len*sizeof(double));

    for (int i = 0; i < window; i++) {
        wsum += arr[i];
        sma[i] = wsum/(i+1);
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= arr[i-window];
        sma[i] = wsum;
    }
    double_inplace_div(sma+window, len-window, (double) window);
    return sma;
}

float* _float_sma(const float* arr, int len, int window) {
    float wsum = 0;
    float* sma = malloc(len*sizeof(float));

    for (int i = 0; i < window; i++) {
        wsum += arr[i];
        sma[i] = wsum/(i+1);
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= arr[i-window];
        sma[i] = wsum;
    }
    float_inplace_div(sma+window, len-window, (float) window);
    return sma;
}

/**
 * Subtract two arrays, store the result in a third array
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
void _double_sub(double *arr1, double *arr2, double *arr3, int len) {
    __m256d v1, v2;
    for (int i = 0; i < len-len%4; i+=4) {
        v1 = _mm256_loadu_pd(&arr1[i]);
        v2 = _mm256_loadu_pd(&arr2[i]);
        _mm256_storeu_pd(&arr3[i], _mm256_sub_pd(v1, v2));
    }
    for (int i = len-len%4; i<len; i++) {
        arr3[i] = arr1[i]-arr2[i];
    }
}

void _float_sub(float* arr1, float* arr2, float* arr3, int len) {
    __m256 v1, v2;
    for (int i = 0; i < len-len%8; i+=8) {
        v1 = _mm256_loadu_ps(&arr1[i]);
        v2 = _mm256_loadu_ps(&arr2[i]);
        _mm256_storeu_ps(&arr3[i], _mm256_sub_ps(v1, v2));
    }
    for (int i = len-len%8; i<len; i++) {
        arr3[i] = arr1[i]-arr2[i];
    }
}
