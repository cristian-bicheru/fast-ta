#pragma once

#include <immintrin.h>
#include <stdlib.h>

/**
 * AVX Absolute Value
 * @param x
 * @param sign_mask
 * @return
 */
inline __m256 abs_ps(__m256 x, __m256 sign_mask) {
    return _mm256_andnot_ps(sign_mask, x);
}

inline __m256d abs_pd(__m256d x, __m256d sign_mask) {
    return _mm256_andnot_pd(sign_mask, x); // !sign_mask & x
}

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

/**
 * Periodic Volatility Sum
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
double* _double_volatility_sum(double *arr1, int period, int len);
float* _float_volatility_sum(float *arr1, int period, int len);

/**
 * Vectorized Array Division
 * @param arr
 * @param len
 * @param x
 */
double* _double_div_arr(double* arr1, double* arr2, int len);
float* _float_div_arr(float* arr1, float* arr2, int len);

/**
 * Vectorized Inplace Multiplication
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_mul(double* arr, int len, double x);
void _float_inplace_mul(float* arr, int len, float x);

/**
 * Vectorized Inplace Addition
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_add(double* arr, int len, double x);
void _float_inplace_add(float* arr, int len, float x);

/**
 * Vectorized Inplace Square
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_square(double* arr, int len);
void _float_inplace_square(float* arr, int len);

/**
 * Vectorized Inplace Absolute Value
 * @param arr
 */
void _double_inplace_abs(double* arr, int len);
void _float_inplace_abs(float* arr, int len);

/**
 * Vectorized Elementwise Inplace Division By Array
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_div_arr(double* arr, int len, double* x);
void _float_inplace_div_arr(float* arr, int len, float* x);

/**
 * Vectorized Max of A Rolling Window Over An Array
 * @param arr
 * @param len
 * @param window
 */
void _double_inplace_running_max(double* arr, int len, int window);
void _float_inplace_running_max(float* arr, int len, int window);

/**
 * Vectorized Min of A Rolling Window Over An Array
 * @param arr
 * @param len
 * @param window
 */
void _double_inplace_running_min(double* arr, int len, int window);
void _float_inplace_running_min(float* arr, int len, int window);
