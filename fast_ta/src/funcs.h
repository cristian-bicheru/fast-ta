#include "Python.h"
#include "numpy/arrayobject.h"

#include <immintrin.h>
#include <stdlib.h>

/**
 * Inplace Exponential Moving Average
 * @param arr
 * @param len
 * @param alpha
 */
void inplace_ema(double* arr, int len, double alpha);

/**
 * Vectorized Pairwise Mean Between Two Arrays
 * @param arr1
 * @param arr2
 * @param len
 * @return
 */
double* _double_pairwise_mean(double* arr1, double* arr2, int len);
float* _float_pairwise_mean(float* arr1, float* arr2, int len);

/**
 * Vectorized Inplace Division
 * @param arr
 * @param len
 * @param x
 */
void double_inplace_div(double* arr, int len, double x);
void float_inplace_div(float* arr, int len, float x);

/**
 * Compute the Simple Moving Average on an array.
 * @param arr
 * @param len
 * @param window
 * @return sma
 */
double* _double_sma(const double* arr, int len, int window);
float* _float_sma(const float* arr, int len, int window);

/**
 * Subtract two arrays, store the result in a third array
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
void _double_sub(double *arr1, double *arr2, double *arr3, int len);
void _float_sub(float* arr1, float* arr2, float* arr3, int len);
