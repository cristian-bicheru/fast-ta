#include <immintrin.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/npy_math.h>
#include <float.h>
#include <stdint.h>

#include "funcs_unaligned.h"
#include "funcs.h"

void _double_pairwise_mean_unaligned(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;
    __double_vector d2 = _double_set1_vec(0.5);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr1[i]);
        v2 = _double_loadu(&arr2[i]);
        v1 = _double_add_vec(v1, v2);
        v1 = _double_mul_vec(v1, d2);
        _double_storeu(&outarr[i], v1);
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = (arr1[i]+arr2[i])/2;
    }
}

void _float_pairwise_mean_unaligned(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;
    __float_vector d2 = _float_set1_vec(0.5f);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr1[i]);
        v2 = _float_loadu(&arr2[i]);
        v1 = _float_add_vec(v1, v2);
        v1 = _float_mul_vec(v1, d2);
        _float_storeu(&outarr[i], v1);
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = (arr1[i]+arr2[i])/2;
    }
}


void _double_div_unaligned(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_div_vec(v, vx));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]/x;
    }
}

void _float_div_unaligned(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_div_vec(v, vx));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]/x;
    }
}

double _double_total_sum_unaligned(const double* arr, int len) {
    __double_vector v;
    __double_vector _sum = _double_setzero_vec();
    double sum;

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _sum = _double_add_vec(_sum, v);
    }

    sum = 0;

    for (int i = 0; i < DOUBLE_VEC_SIZE; i++) {
        sum += _double_index_vec(_sum, i);
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        sum += arr[i];
    }

    return sum;
}

float _float_total_sum_unaligned(const float* arr, int len) {
    __float_vector v;
    __float_vector _sum = _float_setzero_vec();
    float sum;

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _sum = _float_add_vec(_sum, v);
    }

    sum = 0;

    for (int i = 0; i < FLOAT_VEC_SIZE; i++) {
        sum += _float_index_vec(_sum, i);
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        sum += arr[i];
    }

    return sum;
}

void
_double_sub_arr_unaligned(const double *arr1, const double *arr2, int len, double *arr3) {
    __double_vector v1;
    __double_vector v2;

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr1[i]);
        v2 = _double_loadu(&arr2[i]);
        _double_storeu(&arr3[i], _double_sub_vec(v1, v2));
    }
    for (int i = double_get_next_index(len, 0); i<len; i++) {
        arr3[i] = arr1[i]-arr2[i];
    }
}

void
_float_sub_arr_unaligned(const float *arr1, const float *arr2, int len, float *arr3) {
    __float_vector v1;
    __float_vector v2;
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr1[i]);
        v2 = _float_loadu(&arr2[i]);
        _float_storeu(&arr3[i], _float_sub_vec(v1, v2));
    }
    for (int i = float_get_next_index(len, 0); i<len; i++) {
        arr3[i] = arr1[i]-arr2[i];
    }
}

void _intrin_fast_double_roc_unaligned(const double* close, double* roc, int len, int n) {
    const __double_vector mpct = _double_set1_vec(100.);
    __double_vector v1;

    int64_t offset = double_next_aligned_pointer(close+n);

    for (int i = n; i <= n+offset; i++) {
        roc[i] = (close[i]/close[i-n]-1.)*100.;
    }

    for (int i = n+offset; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&close[i-n]);
        _double_storeu(&roc[i], _double_mul_vec(mpct,
                                               _double_div_vec(_double_sub_vec(_double_loadu(&close[i]), v1), v1)));
    }
    for (int i = double_get_next_index(len, n); i<len; i++) {
        roc[i] = (close[i]/close[i-n]-1.)*100.;
    }
}

void _intrin_fast_float_roc_unaligned(const float* close, float* roc, int len, int n) {
    const __float_vector mpct = _float_set1_vec(100.f);
    __float_vector v1;

    int64_t offset = float_next_aligned_pointer(close+n);

    for (int i = n; i <= n+offset; i++) {
        roc[i] = (close[i]/close[i-n]-1.f)*100.f;
    }

    for (int i = n+offset; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&close[i-n]);
        _float_storeu(&roc[i], _float_mul_vec(mpct,
                                             _float_div_vec(_float_sub_vec(_float_loadu(&close[i]), v1), v1)));
    }
    for (int i = float_get_next_index(len, n); i<len; i++) {
        roc[i] = (close[i]/close[i-n]-1.f)*100.f;
    }
}

void _double_volatility_sum_unaligned(const double *arr1, int period, int len, double* outarr) {
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

void _float_volatility_sum_unaligned(const float* arr1, int period, int len, float* outarr) {
    float running_sum = 0;

    for (int i = 1; i < period+1; i++) {
        running_sum += fabsf(arr1[i]-arr1[i-1]);
    }
    outarr[0] = running_sum;

    for (int i = period+1; i < len; i++) {
        running_sum += fabsf(arr1[i]-arr1[i-1]);
        running_sum -= fabsf(arr1[i-period]-arr1[i-period-1]);

        outarr[i-period] = running_sum;
    }
}

void _double_div_arr_unaligned(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr1[i]);
        v2 = _double_loadu(&arr2[i]);
        _double_storeu(&outarr[i], _double_div_vec(v1, v2));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]/arr2[i];
    }
}

void _float_div_arr_unaligned(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr1[i]);
        v2 = _float_loadu(&arr2[i]);
        _float_storeu(&outarr[i], _float_div_vec(v1, v2));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]/arr2[i];
    }
}

void _double_mul_unaligned(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_mul_vec(v, vx));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]*x;
    }
}

void _float_mul_unaligned(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_mul_vec(v, vx));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]*x;
    }
}

void _double_add_unaligned(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_add_vec(v, vx));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]+x;
    }
}

void _float_add_unaligned(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_add_vec(v, vx));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]+x;
    }
}

void _double_sub_unaligned(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_sub_vec(v, vx));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]-x;
    }
}

void _float_sub_unaligned(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_sub_vec(v, vx));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]-x;
    }
}

void _double_square_unaligned(const double* arr, int len, double* outarr) {
    __double_vector v;

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_mul_vec(v, v));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]*arr[i];
    }
}

void _float_square_unaligned(const float* arr, int len, float* outarr) {
    __float_vector v;

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_mul_vec(v, v));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]*arr[i];
    }
}

void _double_abs_unaligned(const double* arr, int len, double* outarr) {
    __double_vector v;
    const __double_vector sign_mask = _double_set1_vec(-0.);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_abs_vec(v, sign_mask));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = fabs(arr[i]);
    }
}

void _float_abs_unaligned(const float* arr, int len, float* outarr) {
    __float_vector v;
    const __float_vector sign_mask = _float_set1_vec(-0.);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_abs_vec(v, sign_mask));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = fabsf(arr[i]);
    }
}

void _double_running_max_unaligned(const double* arr, int len, int window, double* outarr) {
    __double_vector v;
    double m = arr[0];

    for (int i = 0; i < window; i++) {
        m = max(m, arr[i]);
        outarr[i] = m;
    }

    for (int i = 0; i <= len-window+1 - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        for (int j = 0; j < window; j++) {
            v = _double_max_vec(v, _double_loadu(&arr[i+j]));
        }
        _double_storeu(&outarr[i+window-1], v);
    }

    for (int i = double_get_next_index(len-window+1, 0); i < len-window+1; i++) {
        m = arr[i];
        for (int j = 0; j < window; j++) {
            m = max(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _float_running_max_unaligned(const float* arr, int len, int window, float* outarr) {
    __float_vector v;
    float m = arr[0];

    for (int i = 0; i < window; i++) {
        m = max(m, arr[i]);
        outarr[i] = m;
    }

    for (int i = 0; i <= len-window+1 - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        for (int j = 0; j < window; j++) {
            v = _float_max_vec(v, _float_loadu(&arr[i+j]));
        }
        _float_storeu(&outarr[i+window-1], v);
    }

    for (int i = float_get_next_index(len-window+1, 0); i < len-window+1; i++) {
        m = arr[i];
        for (int j = 0; j < window; j++) {
            m = max(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _double_running_min_unaligned(const double* arr, int len, int window, double* outarr) {
    __double_vector v;
    double m = arr[0];

    for (int i = 0; i < window; i++) {
        m = min(m, arr[i]);
        outarr[i] = m;
    }

    for (int i = 0; i <= len-window+1 - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        for (int j = 0; j < window; j++) {
            v = _double_min_vec(v, _double_loadu(&arr[i+j]));
        }
        _double_storeu(&outarr[i+window-1], v);
    }

    for (int i = double_get_next_index(len-window+1, 0); i < len-window+1; i++) {
        m = arr[i];
        for (int j = 0; j < window; j++) {
            m = min(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _float_running_min_unaligned(const float* arr, int len, int window, float* outarr) {
    __float_vector v;
    float m = arr[0];

    for (int i = 0; i < window; i++) {
        m = min(m, arr[i]);
        outarr[i] = m;
    }

    for (int i = 0; i <= len-window+1 - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        for (int j = 0; j < window; j++) {
            v = _float_min_vec(v, _float_loadu(&arr[i+j]));
        }
        _float_storeu(&outarr[i+window-1], v);
    }

    for (int i = float_get_next_index(len-window+1, 0); i < len-window+1; i++) {
        m = arr[i];
        for (int j = 0; j < window; j++) {
            m = min(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _double_set_nan_unaligned(double* arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = NPY_NAN;
    }
}

void _float_set_nan_unaligned(float* arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = NPY_NANF;
    }
}

void _double_consecutive_diff_unaligned(const double* arr, int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;

    for (int i = 1; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr[i]);
        v2 = _double_loadu(&arr[i-1]);
        _double_storeu(&outarr[i], _double_sub_vec(v1, v2));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]-arr[i-1];
    }
}

void _float_consecutive_diff_unaligned(const float* arr, int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;

    for (int i = 1; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr[i]);
        v2 = _float_loadu(&arr[i-1]);
        _float_storeu(&outarr[i], _float_sub_vec(v1, v2));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr[i]-arr[i-1];
    }
}

#if (DOUBLE_VEC_SIZE >= 4)
void _double_tsi_fast_ema_unaligned(double* pc, double* apc, int len, int r, int s) {
    __double_vector v;
    const double alpha1 = 2./(r+1);
    const double alpha2 = 2./(s+1);
    double temp[DOUBLE_VEC_SIZE] = {0};

    temp[0] = alpha2;
    temp[1] = alpha1;
    temp[2] = alpha2;
    temp[3] = alpha1;
    const __double_vector a = _double_loadu(temp);

    temp[0] = 1.-alpha2;
    temp[1] = 1.-alpha1;
    temp[2] = 1.-alpha2;
    temp[3] = 1.-alpha1;
    const __double_vector am = _double_loadu(temp);

    temp[0] = pc[0];
    temp[1] = pc[0];
    temp[2] = apc[0];
    temp[3] = apc[0];
    __double_vector ema = _double_loadu(temp);

    for (int i = 1; i < len; i++) {
        v = _double_loadu2(&pc[i-1], &apc[i-1]);
        v = _double_mul_vec(v, a);
        ema = _double_mul_vec(ema, am);
        ema = _double_add_vec(v, ema);

        pc[i-1] = ema[0];
        pc[i] = ema[1];
        apc[i-1] = ema[2];
        apc[i] = ema[3];
    }

    pc[len-1] = pc[len-2]*(1.-alpha2)+pc[len-1]*(alpha2);
    apc[len-1] = apc[len-2]*(1.-alpha2)+apc[len-1]*(alpha2);
}

void _float_tsi_fast_ema_unaligned(float* pc, float* apc, int len, int r, int s) {
    __double_vector v;
    const double alpha1 = 2./(r+1);
    const double alpha2 = 2./(s+1);
    double temp[DOUBLE_VEC_SIZE] = {0};

    temp[0] = alpha2;
    temp[1] = alpha1;
    temp[2] = alpha2;
    temp[3] = alpha1;
    const __double_vector a = _double_loadu(temp);

    temp[0] = 1.-alpha2;
    temp[1] = 1.-alpha1;
    temp[2] = 1.-alpha2;
    temp[3] = 1.-alpha1;
    const __double_vector am = _double_loadu(temp);

    temp[0] = pc[0];
    temp[1] = pc[0];
    temp[2] = apc[0];
    temp[3] = apc[0];
    __double_vector ema = _double_loadu(temp);

    for (int i = 1; i < len; i++) {
        v = _double_loadu2((double[2]) {pc[i-1], pc[i]}, (double[2]) {apc[i-1], apc[i]});
        v = _double_mul_vec(v, a);
        ema = _double_mul_vec(ema, am);
        ema = _double_add_vec(v, ema);

        pc[i-1] = ema[0];
        pc[i] = ema[1];
        apc[i-1] = ema[2];
        apc[i] = ema[3];
    }

    pc[len-1] = pc[len-2]*(1.-alpha2)+pc[len-1]*(alpha2);
    apc[len-1] = apc[len-2]*(1.-alpha2)+apc[len-1]*(alpha2);
}
#else
void _double_tsi_fast_ema_unaligned(double* pc, double* apc, int len, int r, int s) {
    double v[4];
    double alpha1 = 2./(r+1);
    double alpha2 = 2./(s+1);

    const double a[4] = {alpha2, alpha1, alpha2, alpha1};
    const double am[4] = {1.f-alpha2, 1.f-alpha1, 1.f-alpha2, 1.f-alpha1};

    double ema[4] = {pc[0], pc[0], apc[0], apc[0]};

    for (int i = 1; i < len; i++) {
        v[0] = pc[i-1];
        v[1] = pc[i];
        v[2] = apc[i-1];
        v[3] = apc[i];

        for (int j = 0; j < 4; j++) {
            v[j] *= a[j];
        }
        for (int j = 0; j < 4; j++) {
            ema[j] *= am[j];
        }
        for (int j = 0; j < 4; j++) {
            ema[j] += v[j];
        }

        pc[i-1] = ema[0];
        pc[i] = ema[1];
        apc[i-1] = ema[2];
        apc[i] = ema[3];
    }

    pc[len-1] = pc[len-2]*(1.-alpha2)+pc[len-1]*(alpha2);
    apc[len-1] = apc[len-2]*(1.-alpha2)+apc[len-1]*(alpha2);
}

void _float_tsi_fast_ema_unaligned(float* pc, float* apc, int len, int r, int s) {
    float v[4];
    float alpha1 = 2./(r+1);
    float alpha2 = 2./(s+1);

    const float a[4] = {alpha2, alpha1, alpha2, alpha1};
    const float am[4] = {1.f-alpha2, 1.f-alpha1, 1.f-alpha2, 1.f-alpha1};

    float ema[4] = {pc[0], pc[0], apc[0], apc[0]};

    for (int i = 1; i < len; i++) {
        v[0] = pc[i-1];
        v[1] = pc[i];
        v[2] = apc[i-1];
        v[3] = apc[i];

        for (int j = 0; j < 4; j++) {
            v[j] *= a[j];
        }
        for (int j = 0; j < 4; j++) {
            ema[j] *= am[j];
        }
        for (int j = 0; j < 4; j++) {
            ema[j] += v[j];
        }

        pc[i-1] = ema[0];
        pc[i] = ema[1];
        apc[i-1] = ema[2];
        apc[i] = ema[3];
    }

    pc[len-1] = pc[len-2]*(1.-alpha2)+pc[len-1]*(alpha2);
    apc[len-1] = apc[len-2]*(1.-alpha2)+apc[len-1]*(alpha2);
}
#endif

void _double_pairwise_max_unaligned(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v;
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_max_vec(_double_loadu(&arr1[i]), _double_loadu(&arr2[i]));
        _double_storeu(&outarr[i], v);
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = max(arr1[i], arr2[i]);
    }
}

void _float_pairwise_max_unaligned(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v;
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_max_vec(_float_loadu(&arr1[i]), _float_loadu(&arr2[i]));
        _float_storeu(&outarr[i], v);
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = max(arr1[i], arr2[i]);
    }
}

void _double_pairwise_min_unaligned(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v;
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_min_vec(_double_loadu(&arr1[i]), _double_loadu(&arr2[i]));
        _double_storeu(&outarr[i], v);
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = min(arr1[i], arr2[i]);
    }
}

void _float_pairwise_min_unaligned(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v;
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_min_vec(_float_loadu(&arr1[i]), _float_loadu(&arr2[i]));
        _float_storeu(&outarr[i], v);
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = min(arr1[i], arr2[i]);
    }
}

void _double_running_sum_unaligned(const double* arr, int len, int window, double* outarr) {
    double wsum = 0;

    for (int i = 0; i < window; i++) {
        wsum += arr[i];
        outarr[i] = wsum;
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= arr[i-window];
        outarr[i] = wsum;
    }
}

void _float_running_sum_unaligned(const float* arr, int len, int window, float* outarr) {
    float wsum = 0;

    for (int i = 0; i < window; i++) {
        wsum += arr[i];
        outarr[i] = wsum;
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= arr[i-window];
        outarr[i] = wsum;
    }
}

void _double_add_arr_unaligned(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v;
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_add_vec(_double_loadu(&arr1[i]), _double_loadu(&arr2[i]));
        _double_storeu(&outarr[i], v);
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]+arr2[i];
    }
}

void _float_add_arr_unaligned(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v;
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_add_vec(_float_loadu(&arr1[i]), _float_loadu(&arr2[i]));
        _float_storeu(&outarr[i], v);
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]+arr2[i];
    }
}

void _double_memcpy_unaligned(const double* src, int len, double* outarr) {
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i], _double_loadu(&src[i]));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = src[i];
    }
}

void _float_memcpy_unaligned(const float* src, int len, float* outarr) {
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i], _float_loadu(&src[i]));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = src[i];
    }
}

void _double_div_diff_unaligned(const double* arr1, const double* arr2, const double* arr3, int len, double* outarr) {
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i], _double_div_vec(_double_loadu(&arr1[i]),
                                                  _double_sub_vec(_double_loadu(&arr2[i]),
                                                                  _double_loadu(&arr3[i]))));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]/(arr2[i]-arr3[i]);
    }
}

void _float_div_diff_unaligned(const float* arr1, const float* arr2, const float* arr3, int len, float* outarr) {
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i], _float_div_vec(_float_loadu(&arr1[i]),
                                                _float_sub_vec(_float_loadu(&arr2[i]),
                                                               _float_loadu(&arr3[i]))));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]/(arr2[i]-arr3[i]);
    }
}

void _double_mul_arr_unaligned(const double* arr1, const double* arr2, int len, double* outarr) {
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                      _double_mul_vec(_double_loadu(&arr1[i]),
                                      _double_loadu(&arr2[i])));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]*arr2[i];
    }
}

void _float_mul_arr_unaligned(const float* arr1, const float* arr2, int len, float* outarr) {
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                     _float_mul_vec(_float_loadu(&arr1[i]),
                                    _float_loadu(&arr2[i])));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = arr1[i]*arr2[i];
    }
}

void _double_cumsum_unaligned(const double* arr1, int len, double* outarr) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr1[i];
        outarr[i] = sum;
    }
}

void _float_cumsum_unaligned(const float* arr1, int len, float* outarr) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr1[i];
        outarr[i] = sum;
    }
}

void _double_running_sum_div_unaligned(const double* arr1, const double* arr2, int len, int window, double* outarr) {
    double wsum1 = 0;
    double wsum2 = 0;
    double* buffer1 = malloc(window*sizeof(double));
    double* buffer2 = malloc(window*sizeof(double));
    int buf_loc = 0;

    for (int i = 0; i < window; i++) {
        wsum1 += arr1[i];
        wsum2 += arr2[i];
        buffer1[i] = arr1[i];
        buffer2[i] = arr2[i];
        outarr[i] = wsum1/wsum2;
    }

    for (int i = window; i < len; i++) {
        wsum1 += arr1[i];
        wsum1 -= buffer1[buf_loc];
        wsum2 += arr2[i];
        wsum2 -= buffer2[buf_loc];

        buffer1[buf_loc] = arr1[i];
        buffer2[buf_loc] = arr2[i];

        buf_loc++;
        if (buf_loc >= window) {
            buf_loc = 0;
        }

        outarr[i] = wsum1/wsum2;
    }

    free(buffer1);
    free(buffer2);
}

void _float_running_sum_div_unaligned(const float* arr1, const float* arr2, int len, int window, float* outarr) {
    float wsum1 = 0;
    float wsum2 = 0;
    float* buffer1 = malloc(window*sizeof(float));
    float* buffer2 = malloc(window*sizeof(float));
    int buf_loc = 0;

    for (int i = 0; i < window; i++) {
        wsum1 += arr1[i];
        wsum2 += arr2[i];
        buffer1[i] = arr1[i];
        buffer2[i] = arr2[i];
        outarr[i] = wsum1/wsum2;
    }

    for (int i = window; i < len; i++) {
        wsum1 += arr1[i];
        wsum1 -= buffer1[buf_loc];
        wsum2 += arr2[i];
        wsum2 -= buffer2[buf_loc];

        buffer1[buf_loc] = arr1[i];
        buffer2[buf_loc] = arr2[i];

        buf_loc++;
        if (buf_loc >= window) {
            buf_loc = 0;
        }

        outarr[i] = wsum1/wsum2;
    }

    free(buffer1);
    free(buffer2);
}

void _double_vec_mean_unaligned(const double** arrm, int n, int len, double* outarrm) {
    __double_vector vsum;
    double sum;
    double sf = 1./n;
    __double_vector s = _double_set1_vec(sf);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        vsum = _double_setzero_vec();
        for (int j = 0; j < n; j++) {
            vsum = _double_add_vec(vsum, _double_loadu(&arrm[j][i]));
        }
        vsum = _double_mul_vec(vsum, s);
        _double_storeu(&outarrm[i], vsum);
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        sum = 0;
        for (int j = 0; j < n; j++) {
            sum += arrm[j][i];
        }
        outarrm[i] = sum*sf;
    }
}

void _float_vec_mean_unaligned(const float** arrm, int n, int len, float* outarrm) {
    __float_vector vsum;
    float sum;
    float sf = 1.f/n;
    __float_vector s = _float_set1_vec(sf);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        vsum = _float_setzero_vec();
        for (int j = 0; j < n; j++) {
            vsum = _float_add_vec(vsum, _float_loadu(&arrm[j][i]));
        }
        vsum = _float_mul_vec(vsum, s);
        _float_storeu(&outarrm[i], vsum);
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        sum = 0;
        for (int j = 0; j < n; j++) {
            sum += arrm[j][i];
        }
        outarrm[i] = sum*sf;
    }
}

void _double_mfi_algo_unaligned(const double* high, const double* low, const double* close, const double* volume, int n, int len, double* outarr) {
    double psum = 0;
    double* pvals = malloc(n*sizeof(double));
    double nsum = 0;
    double* nvals = malloc(n*sizeof(double));
    double t;
    double lt;
    int index = 0;

    const double* arrm[3] = {high+n, low+n, close+n};
    _double_vec_mean_unaligned(arrm, 3, len-n, outarr);

    lt = (high[0]+low[0]+close[0])/3;
    for (int i = 1; i < n+1; i++) {
        t = (high[i]+low[i]+close[i])/3;
        if (t > lt) {
            lt = t;
            t *= volume[i];
            psum += t;
            pvals[i-1] = t;
            nvals[i-1] = 0;
        } else {
            lt = t;
            t *= volume[i];
            nsum += t;
            pvals[i-1] = 0;
            nvals[i-1] = t;
        }
    }
    outarr[0] = psum/nsum;

    for (int i = 1; i < len-n; i++) {
        t = outarr[i];

        psum -= pvals[index];
        nsum -= nvals[index];

        if (t > lt) {
            lt = t;
            t *= volume[i+n];
            psum += t;
            pvals[index] = t;
            nvals[index] = 0;
        } else {
            lt = t;
            t *= volume[i+n];
            nsum += t;
            pvals[index] = 0;
            nvals[index] = t;
        }

        index++;
        if (index >= n) {
            index = 0;
        }

        outarr[i] = psum/nsum;
    }

    free(pvals);
    free(nvals);
}

void _float_mfi_algo_unaligned(const float* high, const float* low, const float* close, const float* volume, int n, int len, float* outarr) {
    float psum = 0;
    float* pvals = malloc(n*sizeof(float));
    float nsum = 0;
    float* nvals = malloc(n*sizeof(float));
    float t;
    float lt;
    int index = 0;

    const float* arrm[3] = {high+n, low+n, close+n};
    _float_vec_mean_unaligned(arrm, 3, len-n, outarr);

    lt = (high[0]+low[0]+close[0])/3;
    for (int i = 1; i < n+1; i++) {
        t = (high[i]+low[i]+close[i])/3;
        if (t > lt) {
            lt = t;
            t *= volume[i];
            psum += t;
            pvals[i-1] = t;
            nvals[i-1] = 0;
        } else {
            lt = t;
            t *= volume[i];
            nsum += t;
            pvals[i-1] = 0;
            nvals[i-1] = t;
        }
    }
    outarr[0] = psum/nsum;

    for (int i = 1; i < len-n; i++) {
        t = outarr[i];

        psum -= pvals[index];
        nsum -= nvals[index];

        if (t > lt) {
            lt = t;
            t *= volume[i+n];
            psum += t;
            pvals[index] = t;
            nvals[index] = 0;
        } else {
            lt = t;
            t *= volume[i+n];
            nsum += t;
            pvals[index] = 0;
            nvals[index] = t;
        }

        index++;
        if (index >= n) {
            index = 0;
        }

        outarr[i] = psum/nsum;
    }

    free(pvals);
    free(nvals);
}

void _double_recip_unaligned(const double* arr1, int len, double* outarr) {
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                      _double_recp_vec(_double_loadu(&arr1[i])));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = 1./arr1[i];
    }
}

void _float_recip_unaligned(const float* arr1, int len, float* outarr) {
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                     _float_recp_vec(_float_loadu(&arr1[i])));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = 1.f/arr1[i];
    }
}

void _double_nvi_unaligned(const double* close, const double* volume, int len, double* outarr) {
    outarr[0] = 1000.;
    for (int i = 1; i < len; i++) {
        if (volume[i] < volume[i-1]) {
            outarr[i] = outarr[i-1]*close[i]/close[i-1];
        } else {
            outarr[i] = outarr[i-1];
        }
    }
}

void _float_nvi_unaligned(const float* close, const float* volume, int len, float* outarr) {
    outarr[0] = 1000.f;
    for (int i = 1; i < len; i++) {
        if (volume[i] < volume[i-1]) {
            outarr[i] = outarr[i-1]*close[i]/close[i-1];
        } else {
            outarr[i] = outarr[i-1];
        }
    }
}

void _double_obv_unaligned(const double* close, const double* volume, int len, double* outarr) {
    outarr[0] = volume[0];
    for (int i = 1; i < len; i++) {
        if (close[i] > close[i-1]) {
            outarr[i] = outarr[i-1]+volume[i];
        } else if (close[i] < close[i-1]) {
            outarr[i] = outarr[i-1]-volume[i];
        } else {
            outarr[i] = outarr[i-1];
        }
    }
}
void _float_obv_unaligned(const float* close, const float* volume, int len, float* outarr) {
    outarr[0] = volume[0];
    for (int i = 1; i < len; i++) {
        if (close[i] > close[i-1]) {
            outarr[i] = outarr[i-1]+volume[i];
        } else if (close[i] < close[i-1]) {
            outarr[i] = outarr[i-1]-volume[i];
        } else {
            outarr[i] = outarr[i-1];
        }
    }
}

void _double_vpt_unaligned(const double* close, const double* volume, int len, double* outarr) {
    outarr[0] = 0;
    for (int i = 1; i < len; i++) {
        outarr[i] = outarr[i-1] + volume[i]*(close[i]/close[i-1]-1);
    }
}

void _float_vpt_unaligned(const float* close, const float* volume, int len, float* outarr) {
    outarr[0] = 0;
    for (int i = 1; i < len; i++) {
        outarr[i] = outarr[i - 1] + volume[i] * (close[i] / close[i - 1] - 1);
    }
}

void _double_vwap_unaligned(const double* high, const double* low, const double* close,
                  const double* volume, int n, int len, double* outarr) {
    double s1 = 0;
    double s2 = 0;
    double* ps1 = malloc(n*sizeof(double));
    double* ps2 = malloc(n*sizeof(double));
    int index = 0;

    const double* arrm[3] = {high, low, close};
    _double_vec_mean_unaligned(arrm, 3, len, outarr);
    _double_mul_arr_unaligned(outarr, volume, len, outarr);

    s1 = _double_total_sum(volume, n);
    s2 = _double_total_sum(outarr, n);
    _double_memcpy_unaligned(volume, n, ps1);
    _double_memcpy_unaligned(outarr, n, ps2);

    outarr[n] = s2/s1;

    for (int i = n+1; i < len; i++) {
        s1 += volume[i] - ps1[index];
        s2 += outarr[i] - ps2[index];

        ps1[index] = volume[i];
        ps2[index] = outarr[i];

        index++;
        if (index >= n) {
            index = 0;
        }

        outarr[i] = s2/s1;
    }

    _double_set_nan_unaligned(outarr, n);

    free(ps1);
    free(ps2);
}

void _float_vwap_unaligned(const float* high, const float* low, const float* close,
                 const float* volume, int n, int len, float* outarr) {
    float s1 = 0;
    float s2 = 0;
    float* ps1 = malloc(n*sizeof(float));
    float* ps2 = malloc(n*sizeof(float));
    int index = 0;

    const float* arrm[3] = {high, low, close};
    _float_vec_mean_unaligned(arrm, 3, len, outarr);
    _float_mul_arr_unaligned(outarr, volume, len, outarr);

    s1 = _float_total_sum(volume, n);
    s2 = _float_total_sum(outarr, n);
    _float_memcpy_unaligned(volume, n, ps1);
    _float_memcpy_unaligned(outarr, n, ps2);

    outarr[n] = s2/s1;

    for (int i = n+1; i < len; i++) {
        s1 += volume[i] - ps1[index];
        s2 += outarr[i] - ps2[index];

        ps1[index] = volume[i];
        ps2[index] = outarr[i];

        index++;
        if (index >= n) {
            index = 0;
        }

        outarr[i] = s2/s1;
    }

    _float_set_nan_unaligned(outarr, n);

    free(ps1);
    free(ps2);
}

void _double_tr_unaligned(const double* high, const double* low, const double* close,
                int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;
    __double_vector v3;
    double n1;
    double n2;
    double n3;
    const __double_vector sign_mask = _double_set1_vec(-0.);

    outarr[0] = high[0]-low[0];

    for (int i = 1; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&high[i]);
        v2 = _double_loadu(&close[i-1]);
        v3 = _double_loadu(&low[i]);
        _double_storeu(&outarr[i],
                      _double_max_vec(
                              _double_max_vec(
                                      _double_abs_vec(_double_sub_vec(v1, v2), sign_mask),
                                      _double_sub_vec(v1, v3)
                              ),
                              _double_abs_vec(_double_sub_vec(v3, v2), sign_mask)
                      )
        );
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        n1 = high[i];
        n2 = close[i-1];
        n3 = low[i];
        outarr[i] = max(max(
                fabs(n1-n2),
                n1-n3
        ),
                        fabs(n3-n2)
        );
    }
}



void _float_tr_unaligned(const float* high, const float* low, const float* close,
               int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;
    __float_vector v3;
    float n1;
    float n2;
    float n3;
    const __float_vector sign_mask = _float_set1_vec(-0.);

    outarr[0] = high[0]-low[0];

    for (int i = 1; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&high[i]);
        v2 = _float_loadu(&close[i-1]);
        v3 = _float_loadu(&low[i]);
        _float_storeu(&outarr[i],
                     _float_max_vec(
                             _float_max_vec(
                                     _float_abs_vec(_float_sub_vec(v1, v2), sign_mask),
                                     _float_sub_vec(v1, v3)
                             ),
                             _float_abs_vec(_float_sub_vec(v3, v2), sign_mask)
                     )
        );
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        n1 = high[i];
        n2 = close[i-1];
        n3 = low[i];
        outarr[i] = max(max(
                fabsf(n1-n2),
                n1-n3
        ),
                        fabsf(n3-n2)
        );
    }
}


void _double_rsqrt_unaligned(const double* arr1, int len, double* outarr) {
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                      _double_rsqrt_vec(_double_loadu(&arr1[i])));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = 1./sqrt(arr1[i]);
    }
}

void _float_rsqrt_unaligned(const float* arr1, int len, float* outarr) {
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                     _float_rsqrt_vec(_float_loadu(&arr1[i])));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = 1.f/sqrtf(arr1[i]);
    }
}

void _double_sqrt_unaligned(const double* arr1, int len, double* outarr) {
    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                      _double_sqrt_vec(_double_loadu(&arr1[i])));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        outarr[i] = sqrt(arr1[i]);
    }
}

void _float_sqrt_unaligned(const float* arr1, int len, float* outarr) {
    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                     _float_sqrt_vec(_float_loadu(&arr1[i])));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        outarr[i] = sqrtf(arr1[i]);
    }
}

void _double_running_stddev_unaligned(const double* arr1, const double* arr2, int len, int n, double* outarr) {
    double* temp1 = malloc(n*sizeof(double));

    for (int i = 0; i < n; i++) {
        _double_sub_unaligned(arr1, i+1, arr2[i], temp1);
        _double_square_unaligned(temp1, i+1, temp1);
        outarr[i] = _double_total_sum(temp1, i+1)/(i+1);
    }

    for (int i = n; i < len; i++) {
        _double_sub_unaligned(arr1+i-n+1, n, arr2[i], temp1);
        _double_square_unaligned(temp1, n, temp1);
        outarr[i] = _double_total_sum(temp1, n);
    }

    _double_div_unaligned(outarr+n, len-n, n, outarr+n);
    _double_sqrt_unaligned(outarr, len, outarr);
    free(temp1);
}

void _float_running_stddev_unaligned(const float* arr1, const float* arr2, int len, int n, float* outarr) {
    float* temp1 = malloc(n*sizeof(float));

    for (int i = 0; i < n; i++) {
        _float_sub_unaligned(arr1, i+1, arr2[i], temp1);
        _float_square_unaligned(temp1, i+1, temp1);
        outarr[i] = _float_total_sum(temp1, i+1)/(i+1);
    }

    for (int i = n; i < len; i++) {
        _float_sub_unaligned(arr1+i-n+1, n, arr2[i], temp1);
        _float_square_unaligned(temp1, n, temp1);
        outarr[i] = _float_total_sum(temp1, n);
    }

    _float_div_unaligned(outarr+n, len-n, n, outarr+n);
    _float_sqrt_unaligned(outarr, len, outarr);
    free(temp1);
}

void _double_er_unaligned(const double* close, int len, int n, double* outarr) {
    double sum = 0;
    __double_vector v;
    const __double_vector sign_mask = _double_set1_vec(-0.);

    outarr[0] = 0;

    for (int i = 1; i < n+1; i++) {
        sum += fabs(close[i]-close[i-1]);
    }

    for (int i = 1; i < n+1; i++) {
        outarr[i] = fabs((close[i]-close[0])/sum);
    }

    for (int i = n+1; i < len; i++) {
        sum += fabs(close[i]-close[i-1]);
        sum -= fabs(close[i-n]-close[i-n-1]);
        outarr[i] = sum;
    }

    int64_t offset = double_next_aligned_pointer(close+n+1);

    for (int i = n+1; i < n+1+offset; i++) {
        outarr[i] = fabs(close[i]-close[i-n])/outarr[i];
    }

    for (int i = n+1+offset; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&close[i]);
        v = _double_abs_vec(_double_sub_vec(v, _double_loadu(&close[i-n])), sign_mask);
        _double_storeu(&outarr[i], _double_div_vec(v, _double_loadu(&outarr[i])));
    }

    for (int i = double_get_next_index(len, n+1); i < len; i++) {
        outarr[i] = fabs(close[i]-close[i-n])/outarr[i];
    }
}

void _float_er_unaligned(const float* close, int len, int n, float* outarr) {
    float sum = 0;
    __float_vector v;
    const __float_vector sign_mask = _float_set1_vec(-0.);

    outarr[0] = 0;

    for (int i = 1; i < n+1; i++) {
        sum += fabsf(close[i]-close[i-1]);
    }

    for (int i = 1; i < n+1; i++) {
        outarr[i] = fabsf((close[i]-close[0])/sum);
    }

    for (int i = n+1; i < len; i++) {
        sum += fabsf(close[i]-close[i-1]);
        sum -= fabsf(close[i-n]-close[i-n-1]);
        outarr[i] = sum;
    }

    int64_t offset = float_next_aligned_pointer(close+n+1);

    for (int i = n+1; i < n+1+offset; i++) {
        outarr[i] = fabsf(close[i]-close[i-n])/outarr[i];
    }

    for (int i = n+1+offset; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&close[i]);
        v = _float_abs_vec(_float_sub_vec(v, _float_loadu(&close[i-n])), sign_mask);
        _float_storeu(&outarr[i], _float_div_vec(v, _float_loadu(&outarr[i])));
    }

    for (int i = float_get_next_index(len, n+1); i < len; i++) {
        outarr[i] = fabsf(close[i]-close[i-n])/outarr[i];
    }
}

void _double_atr_unaligned(const double* high, const double* low, const double* close,
                 int len, int n, double* outarr) {
    _double_tr_unaligned(high, low, close, len, outarr);
    _double_ema(outarr, len, 1./(double) n, outarr);
}

void _float_atr_unaligned(const float* high, const float* low, const float* close,
                int len, int n, float* outarr) {
    _float_tr_unaligned(high, low, close, len, outarr);
    _float_ema(outarr, len, 1.f/(float) n, outarr);
}