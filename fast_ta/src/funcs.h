#pragma once

#include <stdlib.h>
#include "generic_simd.h"

/**
 * Exponential Moving Average
 * @param arr
 * @param len
 * @param alpha
 */
void _double_ema(const double* arr, int len, double alpha, double* outarr);
void _float_ema(const float* arr, int len, float alpha, float* outarr);

/**
 * Vectorized Pairwise Mean Between Two Arrays
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_pairwise_mean(const double* arr1, const double* arr2, int len, double* outarr);
void _float_pairwise_mean(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Vectorized Division By Constant
 * @param arr
 * @param len
 * @param x
 * @param outarr
 */
void _double_div(const double* arr, int len, double x, double* outarr);
void _float_div(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Sum of Array
 * @param arr
 * @param len
 * @return sum
 */
double _double_cumilative_sum(const double* arr, int len);
float _float_cumilative_sum(const float* arr, int len);

/**
 * Compute the Simple Moving Average on an array.
 * @param arr
 * @param len
 * @param window
 * @return sma
 */
void _double_sma(const double* arr, int len, int window, double* outarr);
void _float_sma(const float* arr, int len, int window, float* outarr);

/**
 * Subtract two arrays, store the result in a third array
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
void _double_sub_arr(const double *arr1, const double *arr2, int len, double *arr3);
void _float_sub_arr(const float *arr1, const float *arr2, int len, float *arr3);

/**
 * Periodic Volatility Sum
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
void _double_volatility_sum(const double *arr1, int period, int len, double* outarr);
void _float_volatility_sum(const float *arr1, int period, int len, float* outarr);

/**
 * Vectorized Array Division
 * @param arr
 * @param len
 * @param x
 */
void _double_div_arr(const double* arr1, const double* arr2, int len, double* outarr);
void _float_div_arr(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Vectorized Inplace Multiplication
 * @param arr
 * @param len
 * @param x
 */
void _double_mul(const double* arr, int len, double x, double* outarr);
void _float_mul(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Inplace Addition
 * @param arr
 * @param len
 * @param x
 */
void _double_add(const double* arr, int len, double x, double* outarr);
void _float_add(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Inplace Square
 * @param arr
 * @param len
 * @param x
 */
void _double_square(const double* arr, int len, double* outarr);
void _float_square(const float* arr, int len, float* outarr);

/**
 * Vectorized Inplace Absolute Value
 * @param arr
 */
void _double_abs(const double* arr, int len, double* outarr);
void _float_abs(const float* arr, int len, float* outarr);

/**
 * Vectorized Max of A Rolling Window Over An Array
 * @param arr
 * @param len
 * @param window
 */
void _double_running_max(const double* arr, int len, int window, double* outarr);
void _float_running_max(const float* arr, int len, int window, float* outarr);

/**
 * Vectorized Min of A Rolling Window Over An Array
 * @param arr
 * @param len
 * @param window
 */
void _double_running_min(const double* arr, int len, int window, double* outarr);
void _float_running_min(const float* arr, int len, int window, float* outarr);

/**
 * Set Values to NaN
 * @param arr
 * @param len
 */
void _double_set_nan(double* arr, int len);
void _float_set_nan(float* arr, int len);

/**
 * Calculate Difference Between Consecutive Elements
 * @param arr
 * @param len
 * @param outarr
 */
void _double_consecutive_diff(const double* arr, int len, double* outarr);
void _float_consecutive_diff(const float* arr, int len, float* outarr);

/**
 * TSI-Specific Vectorized EMA Algorithm
 * @param pc
 * @param apc
 * @param len
 * @param r
 * @param s
 */
void _double_tsi_fast_ema(double* pc, double* apc, int len, int r, int s);
void _float_tsi_fast_ema(float* pc, float* apc, int len, int r, int s);

/**
 * Vectorized Pairwise Min/Max
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_pairwise_max(const double* arr1, const double* arr2, int len, double* outarr);
void _float_pairwise_max(const float* arr1, const float* arr2, int len, float* outarr);
void _double_pairwise_min(const double* arr1, const double* arr2, int len, double* outarr);
void _float_pairwise_min(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Running Sum
 * @param arr
 * @param len
 * @param window
 * @param outarr
 */
void _double_running_sum(const double* arr, int len, int window, double* outarr);
void _float_running_sum(const float* arr, int len, int window, float* outarr);

/**
 * Vectorized Array Addition
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_add_arr(const double* arr1, const double* arr2, int len, double* outarr);
void _float_add_arr(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Vectorized memcpy
 * @param src
 * @param len
 * @param outarr
 */
void _double_memcpy(const double* src, int len, double* outarr);
void _float_memcpy(const float* src, int len, float* outarr);

/**
 * Vectorized Division of Array By Difference of Two Other Arrays
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_div_diff(const double* arr1, const double* arr2, const double* arr3, int len, double* outarr);
void _float_div_diff(const float* arr1, const float* arr2, const float* arr3, int len, float* outarr);

/**
 * Vectorized Elementwise Array Multiplication
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_mul_arr(const double* arr1, const double* arr2, int len, double* outarr);
void _float_mul_arr(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Cumulative Sum over Array
 * @param arr1
 * @param len
 * @param outarr
 */
void _double_cumsum(const double* arr1, int len, double* outarr);
void _float_cumsum(const float* arr1, int len, float* outarr);
