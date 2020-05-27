#pragma once

#include <stdlib.h>
#include "../generic_simd.h"

void _intrin_fast_double_roc_unaligned(const double* close, double* roc, int len, int n);
void _intrin_fast_float_roc_unaligned(const float* close, float* roc, int len, int n);
/**
 * Create Vector Mask As Described In Issue #18
 * @param len
 * @return
 */
__int_vector _double_create_mask_unaligned(int len);
__int_vector _float_create_mask_unaligned(int len);

/**
 * Vectorized Pairwise Mean Between Two Arrays
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_pairwise_mean_unaligned(const double* arr1, const double* arr2, int len, double* outarr);
void _float_pairwise_mean_unaligned(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Vectorized Division By Constant
 * @param arr
 * @param len
 * @param x
 * @param outarr
 */
void _double_div_unaligned(const double* arr, int len, double x, double* outarr);
void _float_div_unaligned(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Sum of Array
 * @param arr
 * @param len
 * @return sum
 */
double _double_total_sum_unaligned(const double* arr, int len);
float _float_total_sum_unaligned(const float* arr, int len);

/**
 * Compute the Simple Moving Average on an array.
 * @param arr
 * @param len
 * @param window
 * @return sma
 */
void _double_sma_unaligned(const double* arr, int len, int window, double* outarr);
void _float_sma_unaligned(const float* arr, int len, int window, float* outarr);

/**
 * Subtract two arrays, store the result in a third array
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
void _double_sub_arr_unaligned(const double *arr1, const double *arr2, int len, double *arr3);
void _float_sub_arr_unaligned(const float *arr1, const float *arr2, int len, float *arr3);


/**
 * Periodic Volatility Sum
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
void _double_volatility_sum_unaligned(const double *arr1, int period, int len, double* outarr);
void _float_volatility_sum_unaligned(const float *arr1, int period, int len, float* outarr);

/**
 * Vectorized Array Division
 * @param arr
 * @param len
 * @param x
 */
void _double_div_arr_unaligned(const double* arr1, const double* arr2, int len, double* outarr);
void _float_div_arr_unaligned(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Vectorized Inplace Multiplication
 * @param arr
 * @param len
 * @param x
 */
void _double_mul_unaligned(const double* arr, int len, double x, double* outarr);
void _float_mul_unaligned(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Inplace Addition
 * @param arr
 * @param len
 * @param x
 */
void _double_add_unaligned(const double* arr, int len, double x, double* outarr);
void _float_add_unaligned(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Inplace Subtraction
 * @param arr
 * @param len
 * @param x
 */
void _double_sub_unaligned(const double* arr, int len, double x, double* outarr);
void _float_sub_unaligned(const float* arr, int len, float x, float* outarr);

/**
 * Vectorized Inplace Square
 * @param arr
 * @param len
 * @param x
 */
void _double_square_unaligned(const double* arr, int len, double* outarr);
void _float_square_unaligned(const float* arr, int len, float* outarr);

/**
 * Vectorized Inplace Absolute Value
 * @param arr
 */
void _double_abs_unaligned(const double* arr, int len, double* outarr);
void _float_abs_unaligned(const float* arr, int len, float* outarr);

/**
 * Vectorized Max of A Rolling Window Over An Array
 * @param arr
 * @param len
 * @param window
 */
void _double_running_max_unaligned(const double* arr, int len, int window, double* outarr);
void _float_running_max_unaligned(const float* arr, int len, int window, float* outarr);

/**
 * Vectorized Min of A Rolling Window Over An Array
 * @param arr
 * @param len
 * @param window
 */
void _double_running_min_unaligned(const double* arr, int len, int window, double* outarr);
void _float_running_min_unaligned(const float* arr, int len, int window, float* outarr);

/**
 * Set Values to NaN
 * @param arr
 * @param len
 */
void _double_set_nan_unaligned(double* arr, int len);
void _float_set_nan_unaligned(float* arr, int len);

/**
 * Calculate Difference Between Consecutive Elements
 * @param arr
 * @param len
 * @param outarr
 */
void _double_consecutive_diff_unaligned(const double* arr, int len, double* outarr);
void _float_consecutive_diff_unaligned(const float* arr, int len, float* outarr);

/**
 * TSI-Specific Vectorized EMA Algorithm
 * @param pc
 * @param apc
 * @param len
 * @param r
 * @param s
 */
void _double_tsi_fast_ema_unaligned(double* pc, double* apc, int len, int r, int s);
void _float_tsi_fast_ema_unaligned(float* pc, float* apc, int len, int r, int s);

/**
 * Vectorized Pairwise Min/Max
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_pairwise_max_unaligned(const double* arr1, const double* arr2, int len, double* outarr);
void _float_pairwise_max_unaligned(const float* arr1, const float* arr2, int len, float* outarr);
void _double_pairwise_min_unaligned(const double* arr1, const double* arr2, int len, double* outarr);
void _float_pairwise_min_unaligned(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Running Sum
 * @param arr
 * @param len
 * @param window
 * @param outarr
 */
void _double_running_sum_unaligned(const double* arr, int len, int window, double* outarr);
void _float_running_sum_unaligned(const float* arr, int len, int window, float* outarr);

/**
 * Vectorized Array Addition
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_add_arr_unaligned(const double* arr1, const double* arr2, int len, double* outarr);
void _float_add_arr_unaligned(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Vectorized memcpy
 * @param src
 * @param len
 * @param outarr
 */
void _double_memcpy_unaligned(const double* src, int len, double* outarr);
void _float_memcpy_unaligned(const float* src, int len, float* outarr);

/**
 * Vectorized Division of Array By Difference of Two Other Arrays
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_div_diff_unaligned(const double* arr1, const double* arr2, const double* arr3, int len, double* outarr);
void _float_div_diff_unaligned(const float* arr1, const float* arr2, const float* arr3, int len, float* outarr);

/**
 * Vectorized Elementwise Array Multiplication
 * @param arr1
 * @param arr2
 * @param len
 * @param outarr
 */
void _double_mul_arr_unaligned(const double* arr1, const double* arr2, int len, double* outarr);
void _float_mul_arr_unaligned(const float* arr1, const float* arr2, int len, float* outarr);

/**
 * Cumulative Sum over Array
 * @param arr1
 * @param len
 * @param outarr
 */
void _double_cumsum_unaligned(const double* arr1, int len, double* outarr);
void _float_cumsum_unaligned(const float* arr1, int len, float* outarr);

/**
 * Divide Two Running Sums And Store Result In Array
 * @param arr
 * @param len
 * @param window
 * @param outarr
 */
void _double_running_sum_div_unaligned(const double* arr1, const double* arr2, int len, int window, double* outarr);
void _float_running_sum_div_unaligned(const float* arr1, const float* arr2, int len, int window, float* outarr);

/**
 * Vectorized Arithmetic Mean Over n Arrays
 * @param arrm
 * @param n
 * @param len
 * @param outarrm
 */
void _double_vec_mean_unaligned(const double** arrm, int n, int len, double* outarrm);
void _float_vec_mean_unaligned(const float** arrm, int n, int len, float* outarrm);

/**
 * Divide Sum of Positive Numbers Within Lookback Period By Sum of Negative Numbers
 * @param arr1
 * @param len
 * @param outarr
 */
void _double_mfi_algo_unaligned(const double* high, const double* low, const double* close,
                      const double* volume, int n, int len, double* outarr);
void _float_mfi_algo_unaligned(const float* high, const float* low, const float* close,
                     const float* volume, int n, int len, float* outarr);

/**
 * Compute Elementwise Reciprocal of arr1 and Store in outarr
 * @param arr1
 * @param len
 * @param outarr
 */
void _double_recip_unaligned(const double* arr1, int len, double* outarr);
void _float_recip_unaligned(const float* arr1, int len, float* outarr);

/**
 * NVI Algorithm
 * @param close
 * @param volume
 * @param len
 * @param outarr
 */
void _double_nvi_unaligned(const double* close, const double* volume, int len, double* outarr);
void _float_nvi_unaligned(const float* close, const float* volume, int len, float* outarr);

/**
 * OBV Algorithm
 * @param close
 * @param len
 * @param outarr
 */
void _double_obv_unaligned(const double* close, const double* volume, int len, double* outarr);
void _float_obv_unaligned(const float* close, const float* volume, int len, float* outarr);

/**
 * VPT Algorithm
 * @param close
 * @param len
 * @param outarr
 */
void _double_vpt_unaligned(const double* close, const double* volume, int len, double* outarr);
void _float_vpt_unaligned(const float* close, const float* volume, int len, float* outarr);

/**
 * VWAP Algorithm
 * @param close
 * @param len
 * @param outarr
 */
void _double_vwap_unaligned(const double* high, const double* low, const double* close,
                  const double* volume, int n, int len, double* outarr);
void _float_vwap_unaligned(const float* high, const float* low, const float* close,
                 const float* volume, int n, int len, float* outarr);

/**
 * Vecotorized True Range
 * @param close
 * @param len
 * @param outarr
 */
void _double_tr_unaligned(const double* high, const double* low, const double* close,
                int len, double* outarr);
void _float_tr_unaligned(const float* high, const float* low, const float* close,
               int len, float* outarr);

/**
 * Vectorized Reciprocal Square Root of Array
 * @param arr1
 * @param len
 * @param outarr
 */
void _double_rsqrt_unaligned(const double* arr1, int len, double* outarr);
void _float_rsqrt_unaligned(const float* arr1, int len, float* outarr);

/**
 * Vectorized Square Root of Array
 * @param arr1
 * @param len
 * @param outarr
 */
void _double_sqrt_unaligned(const double* arr1, int len, double* outarr);
void _float_sqrt_unaligned(const float* arr1, int len, float* outarr);

/**
 * Rolling Standard Deviation
 * @param arr1
 * @param arr2
 * @param len
 * @param n
 * @param outarr
 */
void _double_running_stddev_unaligned(const double* arr1, const double* arr2, int len, int n, double* outarr);
void _float_running_stddev_unaligned(const float* arr1, const float* arr2, int len, int n, float* outarr);

/**
 * Vectorized Efficiency Ratio For KAMA
 * @param close
 * @param len
 * @param n
 * @param outarr
 */
void _double_er_unaligned(const double* close, int len, int n, double* outarr);
void _float_er_unaligned(const float* close, int len, int n, float* outarr);

/**
 * Compute Average True Range
 * @param high
 * @param low
 * @param close
 * @param len
 * @param n
 * @param outarr
 */
void _double_atr_unaligned(const double* high, const double* low, const double* close,
                 int len, int n, double* outarr);
void _float_atr_unaligned(const float* high, const float* low, const float* close,
                int len, int n, float* outarr);