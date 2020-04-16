#include <immintrin.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/npy_math.h>

#include "debug_tools.c"
#include "funcs.h"


#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

// TODO: Replace inplace functions with functions that specify a storage destination.

void _double_ema(const double* arr, int len, double alpha, double* outarr) {
    for (int i = 1; i < len; i++) {
        outarr[i] = arr[i-1] * (1-alpha) + arr[i] * alpha;
    }
}

void _float_ema(const float* arr, int len, float alpha, float* outarr) {
    for (int i = 1; i < len; i++) {
        outarr[i] = arr[i-1] * (1-alpha) + arr[i] * alpha;
    }
}

void _double_pairwise_mean(const double* arr1, const double* arr2, int len, double* outarr) {
    __m256d v1;
    __m256d v2;
    __m256d d2 = _mm256_set_pd(0.5, 0.5, 0.5, 0.5);

    for (int i = 0; i < len-len%4; i += 4) {
        v1 = _mm256_loadu_pd(&arr1[i]);
        v2 = _mm256_loadu_pd(&arr2[i]);
        v1 = _mm256_add_pd(v1, v2);
        v1 = _mm256_mul_pd(v1, d2);
        _mm256_storeu_pd(&outarr[i], v1);
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = (arr1[i]+arr2[i])/2;
    }
}

void _float_pairwise_mean(const float* arr1, const float* arr2, int len, float* outarr) {
    __m256 v1;
    __m256 v2;
    __m256 d2 = _mm256_set_ps(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);

    for (int i = 0; i < len-len%8; i += 8) {
        v1 = _mm256_loadu_ps(&arr1[i]);
        v2 = _mm256_loadu_ps(&arr2[i]);
        v1 = _mm256_add_ps(v1, v2);
        v1 = _mm256_mul_ps(v1, d2);
        _mm256_storeu_ps(&outarr[i], v1);
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = (arr1[i]+arr2[i])/2;
    }
}


void _double_div(const double* arr, int len, double x, double* outarr) {
    __m256d v;
    __m256d vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&outarr[i], _mm256_div_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = arr[i]/x;
    }
}

void _float_div(const float* arr, int len, float x, float* outarr) {
    __m256 v;
    __m256 vx;
    vx = _mm256_set_ps(x, x, x, x, x, x, x, x);

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&outarr[i], _mm256_div_ps(v, vx));
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = arr[i]/x;
    }
}

double _double_cumilative_sum(const double* arr, int len) {
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

float _float_cumilative_sum(const float* arr, int len) {
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

void _double_sma(const double* arr, int len, int window, double* outarr) {
    double wsum = 0;

    for (int i = 0; i < window; i++) {
        wsum += arr[i];
        outarr[i] = wsum/(i+1);
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= arr[i-window];
        outarr[i] = wsum;
    }

    _double_div(outarr + window, len - window, (double) window, outarr + window);
}

void _float_sma(const float* arr, int len, int window, float* outarr) {
    float wsum = 0;

    for (int i = 0; i < window; i++) {
        wsum += arr[i];
        outarr[i] = wsum/(i+1);
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= arr[i-window];
        outarr[i] = wsum;
    }

    _float_div(outarr + window, len - window, (float) window, outarr + window);
}

void _double_sub(const double *arr1, const double *arr2, double *arr3, int len) {
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

void _float_sub(const float* arr1, const float* arr2, float* arr3, int len) {
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

void _double_volatility_sum(const double *arr1, int period, int len, double* outarr) {
    double running_sum = 0;

    for (int i = 1; i < period+1; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
    }
    outarr[0] = running_sum;

    for (int i = period+1; i < len; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
        running_sum -= fabs(arr1[i-period]-arr1[i-period-1]);

        outarr[i-period] = running_sum;
    }
}

void _float_volatility_sum(const float* arr1, int period, int len, float* outarr) {
    float running_sum = 0;

    for (int i = 1; i < period+1; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
    }
    outarr[0] = running_sum;

    for (int i = period+1; i < len; i++) {
        running_sum += fabs(arr1[i]-arr1[i-1]);
        running_sum -= fabs(arr1[i-period]-arr1[i-period-1]);

        outarr[i-period] = running_sum;
    }
}

void _double_div_arr(const double* arr1, const double* arr2, int len, double* outarr) {
    __m256d v1;
    __m256d v2;

    for (int i = 0; i < len-len%4; i += 4) {
        v1 = _mm256_loadu_pd(&arr1[i]);
        v2 = _mm256_loadu_pd(&arr2[i]);
        _mm256_storeu_pd(&outarr[i], _mm256_div_pd(v1, v2));
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = arr1[i]/arr2[i];
    }
}

void _float_div_arr(const float* arr1, const float* arr2, int len, float* outarr) {
    __m256 v1;
    __m256 v2;

    for (int i = 0; i < len-len%8; i += 8) {
        v1 = _mm256_loadu_ps(&arr1[i]);
        v2 = _mm256_loadu_ps(&arr2[i]);
        _mm256_storeu_ps(&outarr[i], _mm256_div_ps(v1, v2));
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = arr1[i]/arr2[i];
    }
}

void _double_mul(const double* arr, int len, double x, double* outarr) {
    __m256d v;
    __m256d vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&outarr[i], _mm256_mul_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = arr[i]*x;
    }
}

void _float_mul(const float* arr, int len, float x, float* outarr) {
    __m256 v;
    __m256 vx;
    vx = _mm256_set_ps(x, x, x, x, x, x, x, x);

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&outarr[i], _mm256_mul_ps(v, vx));
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = arr[i]*x;
    }
}

void _double_add(const double* arr, int len, double x, double* outarr) {
    __m256d v;
    __m256d vx;
    vx = _mm256_set_pd(x, x, x, x);

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&outarr[i], _mm256_add_pd(v, vx));
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = arr[i]+x;
    }
}

void _float_add(const float* arr, int len, float x, float* outarr) {
    __m256 v;
    __m256 vx;
    vx = _mm256_set_ps(x, x, x, x, x, x, x, x);

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&outarr[i], _mm256_add_ps(v, vx));
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = arr[i]+x;
    }
}

void _double_square(const double* arr, int len, double* outarr) {
    __m256d v;

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&outarr[i], _mm256_mul_pd(v, v));
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = arr[i]*arr[i];
    }
}

void _float_square(const float* arr, int len, float* outarr) {
    __m256 v;

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&outarr[i], _mm256_mul_ps(v, v));
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = arr[i]*arr[i];
    }
}

void _double_abs(const double* arr, int len, double* outarr) {
    __m256d v;
    const __m256d sign_mask = _mm256_set1_pd(-0.); // -0. = 1 << 63

    for (int i = 0; i < len-len%4; i += 4) {
        v = _mm256_loadu_pd(&arr[i]);
        _mm256_storeu_pd(&outarr[i], abs_pd(v, sign_mask));
    }

    for (int i = len-len%4; i < len; i++) {
        outarr[i] = fabs(arr[i]);
    }
}

void _float_abs(const float* arr, int len, float* outarr) {
    __m256 v;
    const __m256 sign_mask = _mm256_set1_ps(-0.); // -0.f = 1 << 31

    for (int i = 0; i < len-len%8; i += 8) {
        v = _mm256_loadu_ps(&arr[i]);
        _mm256_storeu_ps(&outarr[i], abs_ps(v, sign_mask));
    }

    for (int i = len-len%8; i < len; i++) {
        outarr[i] = fabs(arr[i]);
    }
}

void _double_running_max(const double* arr, int len, int window, double* outarr) {
    __m256d v;
    double m;
    for (int i = 0; i < len-window+1; i++) {
        v = _mm256_loadu_pd(&arr[i]);
        for (int j = 4; j < window-window%4; j += 4) {
            v = _mm256_max_pd(v, _mm256_loadu_pd(&arr[i+j]));
        }
        m = max(max(max(v[0], v[1]), v[2]), v[3]);

        for (int j = window-window%4; j < window; j++) {
            m = max(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _float_running_max(const float* arr, int len, int window, float* outarr) {
    __m256 v;
    float m;
    for (int i = 0; i < len-window+1; i++) {
        v = _mm256_loadu_ps(&arr[i]);
        for (int j = 8; j < window-window%8; j += 8) {
            v = _mm256_max_ps(v, _mm256_loadu_ps(&arr[i+j]));
        }
        m = max(max(max(max(v[0], v[1]), v[2]), v[3]),
                max(max(max(v[4], v[5]), v[6]), v[7]));

        for (int j = window-window%8; j < window; j++) {
            m = max(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _double_running_min(const double* arr, int len, int window, double* outarr) {
    __m256d v;
    double m;
    for (int i = 0; i < len-window+1; i++) {
        v = _mm256_loadu_pd(&arr[i]);
        for (int j = 4; j < window-window%4; j += 4) {
            v = _mm256_min_pd(v, _mm256_loadu_pd(&arr[i+j]));
        }
        m = min(min(min(v[0], v[1]), v[2]), v[3]);

        for (int j = window-window%4; j < window; j++) {
            m = min(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _float_running_min(const float* arr, int len, int window, float* outarr) {
    __m256 v;
    float m;
    for (int i = 0; i < len-window+1; i++) {
        v = _mm256_loadu_ps(&arr[i]);
        for (int j = 8; j < window-window%8; j += 8) {
            v = _mm256_min_ps(v, _mm256_loadu_ps(&arr[i+j]));
        }
        m = min(min(min(min(v[0], v[1]), v[2]), v[3]),
                min(min(min(v[4], v[5]), v[6]), v[7]));

        for (int j = window-window%8; j < window; j++) {
            m = min(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _double_set_nan(double* arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = NPY_NAN;
    }
}

void _float_set_nan(float* arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = NPY_NANF;
    }
}