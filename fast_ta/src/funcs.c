#include <immintrin.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include "debug_tools.c"

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

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
    __m256d v1;
    __m256d v2;
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
    __m256 v1;
    __m256 v2;
    __m256 d2 = _mm256_set_ps(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);

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
void _double_inplace_div(double* arr, int len, double x) {
    __m256d v;
    __m256d vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&arr[i], _mm256_div_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        arr[i] = arr[i]/x;
    }
}

void _float_inplace_div(float* arr, int len, float x) {
    __m256 v;
    __m256 vx;
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
 * Vectorized Cumilative Sum
 * @param arr
 * @param len
 * @return sum
 */
double _double_cumilative_sum(double* arr, int len) {
    __m256d v;
    __m256d _sum = _mm256_set_pd(0,0,0,0);
    double sum;

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _sum = _mm256_add_pd(_sum, v);
    }

    sum = _sum[0] + _sum[1] + _sum[2] + _sum[3];

    for (int i = len-len%4; i < len; i++) {
        sum += arr[i];
    }

    return sum;
}

float _float_cumilative_sum(float* arr, int len) {
    __m256 v;
    __m256 _sum = _mm256_set_ps(0,0,0,0,0,0,0,0);
    float sum;

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _sum = _mm256_add_ps(_sum, v);
    }

    sum =   _sum[0] + _sum[1] + _sum[2] + _sum[3] +
            _sum[4] + _sum[5] + _sum[6] + _sum[7];

    for (int i = len-len%8; i < len; i++) {
        sum += arr[i];
    }

    return sum;
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
    _double_inplace_div(sma + window, len - window, (double) window);
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
    _float_inplace_div(sma + window, len - window, (float) window);
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
    __m256d v1;
    __m256d v2;
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
    __m256 v1;
    __m256 v2;
    for (int i = 0; i < len-len%8; i+=8) {
        v1 = _mm256_loadu_ps(&arr1[i]);
        v2 = _mm256_loadu_ps(&arr2[i]);
        _mm256_storeu_ps(&arr3[i], _mm256_sub_ps(v1, v2));
    }
    for (int i = len-len%8; i<len; i++) {
        arr3[i] = arr1[i]-arr2[i];
    }
}

/**
 * Periodic Volatility Sum
 * @param arr1
 * @param arr2
 * @param arr3
 * @param len
 */
double* _double_volatility_sum(double *arr1, int period, int len) {
    double* vol_sum = malloc((len-period)*sizeof(double));
    double running_sum = 0;

    for (int i = 1; i < period+1; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
    }
    vol_sum[0] = running_sum;

    for (int i = period+1; i < len; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
        running_sum -= fabs(arr1[i-period]-arr1[i-period-1]);

        vol_sum[i-period] = running_sum;
    }

    return vol_sum;
}

float* _float_volatility_sum(float *arr1, int period, int len) {
    float* vol_sum = malloc((len-period)*sizeof(float));
    float running_sum = 0;

    for (int i = 1; i < period+1; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
    }
    vol_sum[0] = running_sum;

    for (int i = period+1; i < len; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
        running_sum -= fabs(arr1[i-period]-arr1[i-period-1]);

        vol_sum[i-period] = running_sum;
    }

    return vol_sum;
}

/**
 * Vectorized Array Division
 * @param arr
 * @param len
 * @param x
 */
double* _double_div_arr(double* arr1, double* arr2, int len) {
    __m256d v1;
    __m256d v2;
    double* ret = malloc(len*sizeof(double));

    for (int i = 0; i < len-len%4; i += 4) {
        v1 = _mm256_loadu_pd(&arr1[i]);
        v2 = _mm256_loadu_pd(&arr2[i]);
        _mm256_storeu_pd(&ret[i], _mm256_div_pd(v1, v2));
    }

    for (int i = len-len%4; i < len; i++) {
        ret[i] = arr1[i]/arr2[i];
    }

    return ret;
}

float* _float_div_arr(float* arr1, float* arr2, int len) {
    __m256 v1;
    __m256 v2;
    float* ret = malloc(len*sizeof(float));

    for (int i = 0; i < len-len%8; i += 8) {
        v1 = _mm256_loadu_ps(&arr1[i]);
        v2 = _mm256_loadu_ps(&arr2[i]);
        _mm256_storeu_ps(&ret[i], _mm256_div_ps(v1, v2));
    }

    for (int i = len-len%8; i < len; i++) {
        ret[i] = arr1[i]/arr2[i];
    }

    return ret;
}

/**
 * Vectorized Inplace Multiplication
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_mul(double* arr, int len, double x) {
    __m256d v;
    __m256d vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&arr[i], _mm256_mul_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        arr[i] = arr[i]*x;
    }
}

void _float_inplace_mul(float* arr, int len, float x) {
    __m256 v;
    __m256 vx;
    vx = _mm256_set_ps(x, x, x, x, x, x, x, x);

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&arr[i], _mm256_mul_ps(v, vx));
    }

    for (int i = len-len%8; i < len; i++) {
        arr[i] = arr[i]*x;
    }
}

/**
 * Vectorized Inplace Addition
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_add(double* arr, int len, double x) {
    __m256d v;
    __m256d vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&arr[i], _mm256_add_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        arr[i] = arr[i]+x;
    }
}

void _float_inplace_add(float* arr, int len, float x) {
    __m256 v;
    __m256 vx;
    vx = _mm256_set_ps(x, x, x, x, x, x, x, x);

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&arr[i], _mm256_add_ps(v, vx));
    }

    for (int i = len-len%8; i < len; i++) {
        arr[i] = arr[i]+x;
    }
}

/**
 * Vectorized Inplace Square
 * @param arr
 * @param len
 * @param x
 */
void _double_inplace_square(double* arr, int len) {
    __m256d v;

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&arr[i], _mm256_mul_pd(v, v));
    }

    for (int i = len-len%4; i < len; i++) {
        arr[i] = arr[i]*arr[i];
    }
}

void _float_inplace_square(float* arr, int len) {
    __m256 v;

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&arr[i], _mm256_mul_ps(v, v));
    }

    for (int i = len-len%8; i < len; i++) {
        arr[i] = arr[i]*arr[i];
    }
}

/**
 * Vectorized Inplace Absolute Value
 * @param arr
 */
void _double_inplace_abs(double* arr, int len) {
    __m256d v;
    const __m256d sign_mask = _mm256_set1_pd(-0.); // -0. = 1 << 63

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&arr[i], abs_pd(v, sign_mask));
    }

    for (int i = len-len%4; i < len; i++) {
        arr[i] = fabs(arr[i]);
    }
}

void _float_inplace_abs(float* arr, int len) {
    __m256 v;
    const __m256 sign_mask = _mm256_set1_ps(-0.); // -0.f = 1 << 31

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&arr[i], abs_ps(v, sign_mask));
    }

    for (int i = len-len%8; i < len; i++) {
        arr[i] = fabs(arr[i]);
    }
}