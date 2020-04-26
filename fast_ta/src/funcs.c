#include <immintrin.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <numpy/npy_math.h>

#include "funcs.h"

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
    __double_vector v1;
    __double_vector v2;
    __double_vector d2 = _double_set1_vec(0.5);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr1[i]);
        v2 = _double_loadu(&arr2[i]);
        v1 = _double_add_vec(v1, v2);
        v1 = _double_mul_vec(v1, d2);
        _double_storeu(&outarr[i], v1);
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = (arr1[i]+arr2[i])/2;
    }
}

void _float_pairwise_mean(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;
    __float_vector d2 = _float_set1_vec(0.5f);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr1[i]);
        v2 = _float_loadu(&arr2[i]);
        v1 = _float_add_vec(v1, v2);
        v1 = _float_mul_vec(v1, d2);
        _float_storeu(&outarr[i], v1);
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = (arr1[i]+arr2[i])/2;
    }
}


void _double_div(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_div_vec(v, vx));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]/x;
    }
}

void _float_div(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_div_vec(v, vx));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]/x;
    }
}

double _double_total_sum(const double* arr, int len) {
    __double_vector v;
    __double_vector _sum = _double_setzero_vec();
    double sum;

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _sum = _double_add_vec(_sum, v);
    }

    sum = 0;

    for (int i = 0; i < DOUBLE_VEC_SIZE; i++) {
        sum += _double_index_vec(_sum, i);
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        sum += arr[i];
    }

    return sum;
}

float _float_total_sum(const float* arr, int len) {
    __float_vector v;
    __float_vector _sum = _float_setzero_vec();
    float sum;

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _sum = _float_add_vec(_sum, v);
    }

    sum = 0;

    for (int i = 0; i < FLOAT_VEC_SIZE; i++) {
        sum += _float_index_vec(_sum, i);
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        sum += arr[i];
    }

    return sum;
}

void _double_sma(const double* arr, int len, int window, double* outarr) {
    double wsum = 0;
    double* temp1 = malloc(window*sizeof(double));
    int index = 0;

    for (int i = 0; i < window; i++) {
        temp1[i] = arr[i];
        wsum += arr[i];
        outarr[i] = wsum/(i+1);
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= temp1[index];

        temp1[index] = arr[i];

        index++;
        if (index >= window) {
            index = 0;
        }

        outarr[i] = wsum;
    }

    _double_div(outarr + window, len - window, (double) window, outarr + window);
    free(temp1);
}

void _float_sma(const float* arr, int len, int window, float* outarr) {
    float wsum = 0;
    float* temp1 = malloc(window*sizeof(float));
    int index = 0;

    for (int i = 0; i < window; i++) {
        temp1[i] = arr[i];
        wsum += arr[i];
        outarr[i] = wsum/(i+1);
    }

    for (int i = window; i < len; i++) {
        wsum += arr[i];
        wsum -= temp1[index];

        temp1[index] = arr[i];

        index++;
        if (index >= window) {
            index = 0;
        }

        outarr[i] = wsum;
    }

    _float_div(outarr + window, len - window, (float) window, outarr + window);
    free(temp1);
}

void
_double_sub_arr(const double *arr1, const double *arr2, int len, double *arr3) {
    __double_vector v1;
    __double_vector v2;
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr1[i]);
        v2 = _double_loadu(&arr2[i]);
        _double_storeu(&arr3[i], _double_sub_vec(v1, v2));
    }
    for (int i = len - len % DOUBLE_VEC_SIZE; i<len; i++) {
        arr3[i] = arr1[i]-arr2[i];
    }
}

void
_float_sub_arr(const float *arr1, const float *arr2, int len, float *arr3) {
    __float_vector v1;
    __float_vector v2;
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr1[i]);
        v2 = _float_loadu(&arr2[i]);
        _float_storeu(&arr3[i], _float_sub_vec(v1, v2));
    }
    for (int i = len - len % FLOAT_VEC_SIZE; i<len; i++) {
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
        running_sum += fabsf(arr1[i]-arr1[i-1]);
    }
    outarr[0] = running_sum;

    for (int i = period+1; i < len; i++) {
        running_sum += fabsf(arr1[i]-arr1[i-1]);
        running_sum -= fabsf(arr1[i-period]-arr1[i-period-1]);

        outarr[i-period] = running_sum;
    }
}

void _double_div_arr(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr1[i]);
        v2 = _double_loadu(&arr2[i]);
        _double_storeu(&outarr[i], _double_div_vec(v1, v2));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]/arr2[i];
    }
}

void _float_div_arr(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr1[i]);
        v2 = _float_loadu(&arr2[i]);
        _float_storeu(&outarr[i], _float_div_vec(v1, v2));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]/arr2[i];
    }
}

void _double_mul(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_mul_vec(v, vx));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]*x;
    }
}

void _float_mul(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_mul_vec(v, vx));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]*x;
    }
}

void _double_add(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_add_vec(v, vx));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]+x;
    }
}

void _float_add(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_add_vec(v, vx));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]+x;
    }
}

void _double_sub(const double* arr, int len, double x, double* outarr) {
    __double_vector v;
    __double_vector vx;
    vx = _double_set1_vec(x);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_sub_vec(v, vx));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]-x;
    }
}

void _float_sub(const float* arr, int len, float x, float* outarr) {
    __float_vector v;
    __float_vector vx;
    vx = _float_set1_vec(x);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_sub_vec(v, vx));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]-x;
    }
}

void _double_square(const double* arr, int len, double* outarr) {
    __double_vector v;

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_mul_vec(v, v));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]*arr[i];
    }
}

void _float_square(const float* arr, int len, float* outarr) {
    __float_vector v;

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_mul_vec(v, v));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr[i]*arr[i];
    }
}

void _double_abs(const double* arr, int len, double* outarr) {
    __double_vector v;
    const __double_vector sign_mask = _double_set1_vec(-0.);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_loadu(&arr[i]);
        _double_storeu(&outarr[i], _double_abs_vec(v, sign_mask));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = fabs(arr[i]);
    }
}

void _float_abs(const float* arr, int len, float* outarr) {
    __float_vector v;
    const __float_vector sign_mask = _float_set1_vec(-0.);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_loadu(&arr[i]);
        _float_storeu(&outarr[i], _float_abs_vec(v, sign_mask));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = fabsf(arr[i]);
    }
}

void _double_running_max(const double* arr, int len, int window, double* outarr) {
    __double_vector v;
    double m;
    for (int i = 0; i < len-window+1; i++) {
        v = _double_loadu(&arr[i]);
        for (int j = DOUBLE_VEC_SIZE; j < window - window % DOUBLE_VEC_SIZE; j += DOUBLE_VEC_SIZE) {
            v = _double_max_vec(v, _double_loadu(&arr[i+j]));
        }

        m = _double_index_vec(v, 0);
        for (int j = 1; j < DOUBLE_VEC_SIZE; j++) {
            m = max(m, _double_index_vec(v, j));
        }

        for (int j = window - window % DOUBLE_VEC_SIZE; j < window; j++) {
            m = max(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _float_running_max(const float* arr, int len, int window, float* outarr) {
    __float_vector v;
    float m;
    for (int i = 0; i < len-window+1; i++) {
        v = _float_loadu(&arr[i]);
        for (int j = FLOAT_VEC_SIZE; j < window - window % FLOAT_VEC_SIZE; j += FLOAT_VEC_SIZE) {
            v = _float_max_vec(v, _float_loadu(&arr[i+j]));
        }

        m = _float_index_vec(v, 0);
        for (int j = 1; j < FLOAT_VEC_SIZE; j++) {
            m = max(m, _float_index_vec(v, j));
        }

        for (int j = window - window % FLOAT_VEC_SIZE; j < window; j++) {
            m = max(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _double_running_min(const double* arr, int len, int window, double* outarr) {
    __double_vector v;
    double m;
    for (int i = 0; i < len-window+1; i++) {
        v = _double_loadu(&arr[i]);
        for (int j = DOUBLE_VEC_SIZE; j < window - window % DOUBLE_VEC_SIZE; j += DOUBLE_VEC_SIZE) {
            v = _double_min_vec(v, _double_loadu(&arr[i+j]));
        }

        m = _double_index_vec(v, 0);
        for (int j = 1; j < DOUBLE_VEC_SIZE; j++) {
            m = min(m, _double_index_vec(v, j));
        }

        for (int j = window - window % DOUBLE_VEC_SIZE; j < window; j++) {
            m = min(m, arr[i+j]);
        }
        outarr[i] = m;
    }
}

void _float_running_min(const float* arr, int len, int window, float* outarr) {
    __float_vector v;
    float m;
    for (int i = 0; i < len-window+1; i++) {
        v = _float_loadu(&arr[i]);
        for (int j = FLOAT_VEC_SIZE; j < window - window % FLOAT_VEC_SIZE; j += FLOAT_VEC_SIZE) {
            v = _float_min_vec(v, _float_loadu(&arr[i+j]));
        }

        m = _float_index_vec(v, 0);
        for (int j = 1; j < FLOAT_VEC_SIZE; j++) {
            m = min(m, _float_index_vec(v, j));
        }

        for (int j = window - window % FLOAT_VEC_SIZE; j < window; j++) {
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

void _double_consecutive_diff(const double* arr, int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;

    for (int i = 0; i < (len-1) - (len-1) % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v1 = _double_loadu(&arr[i]);
        v2 = _double_loadu(&arr[i+1]);
        _double_storeu(&outarr[i], _double_sub_vec(v2, v1));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len-1; i++) {
        outarr[i] = arr[i+1]-arr[i];
    }
}

void _float_consecutive_diff(const float* arr, int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;

    for (int i = 0; i < (len-1) - (len-1) % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v1 = _float_loadu(&arr[i]);
        v2 = _float_loadu(&arr[i+1]);
        _float_storeu(&outarr[i], _float_sub_vec(v2, v1));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len-1; i++) {
        outarr[i] = arr[i+1]-arr[i];
    }
}

#if (DOUBLE_VEC_SIZE >= 4)
void _double_tsi_fast_ema(double* pc, double* apc, int len, int r, int s) {
    __double_vector v;
    double alpha1 = 2./(r+1);
    double alpha2 = 2./(s+1);

    const __double_vector a = _double_set_vec(alpha2, alpha1, alpha2, alpha1);
    const __double_vector am = _double_set_vec(1.-alpha2, 1.-alpha1, 1.-alpha2, 1.-alpha1);

    __double_vector ema = _double_loadu2(&pc[0], &apc[0]);

    for (int i = 1; i < len-1; i++) {
        v = _double_loadu2(&pc[i-1], &apc[i-1]);
        v = _double_mul_vec(v, a);
        ema = _double_mul_vec(ema, am);
        ema = _double_add_vec(v, ema);

        pc[i-1] = ema[0];
        pc[i] = ema[1];
        apc[i-1] = ema[2];
        apc[i] = ema[3];
    }
}

void _float_tsi_fast_ema(float* pc, float* apc, int len, int r, int s) {
    __double_vector v;
    double alpha1 = 2./(r+1);
    double alpha2 = 2./(s+1);

    const __double_vector a = _double_set_vec(alpha2, alpha1, alpha2, alpha1);
    const __double_vector am = _double_set_vec(1.f-alpha2, 1.f-alpha1, 1.f-alpha2, 1.f-alpha1);

    __double_vector ema = _double_loadu2_from_float(&pc[0],&apc[0]);

    for (int i = 1; i < len-1; i++) {
        v = _double_loadu2_from_float(&pc[i-1],&apc[i-1]);
        v = _double_mul_vec(v, a);
        ema = _double_mul_vec(ema, am);
        ema = _double_add_vec(v, ema);

        pc[i-1] = ema[0];
        pc[i] = ema[1];
        apc[i-1] = ema[2];
        apc[i] = ema[3];
    }
}
#else
void _double_tsi_fast_ema(double* pc, double* apc, int len, int r, int s) {
    double v[4];
    double alpha1 = 2./(r+1);
    double alpha2 = 2./(s+1);

    const double a[4] = {alpha2, alpha1, alpha2, alpha1};
    const double am[4] = {1.f-alpha2, 1.f-alpha1, 1.f-alpha2, 1.f-alpha1};

    double ema[4] = {pc[0], pc[1], apc[0], apc[1]};

    for (int i = 1; i < len-1; i++) {
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
}

void _float_tsi_fast_ema(float* pc, float* apc, int len, int r, int s) {
    float v[4];
    float alpha1 = 2./(r+1);
    float alpha2 = 2./(s+1);

    const float a[4] = {alpha2, alpha1, alpha2, alpha1};
    const float am[4] = {1.f-alpha2, 1.f-alpha1, 1.f-alpha2, 1.f-alpha1};

    float ema[4] = {pc[0], pc[1], apc[0], apc[1]};

    for (int i = 1; i < len-1; i++) {
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
}
#endif

void _double_pairwise_max(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v;
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_max_vec(_double_loadu(&arr1[i]), _double_loadu(&arr2[i]));
        _double_storeu(&outarr[i], v);
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = max(arr1[i], arr2[i]);
    }
}

void _float_pairwise_max(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v;
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_max_vec(_float_loadu(&arr1[i]), _float_loadu(&arr2[i]));
        _float_storeu(&outarr[i], v);
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = max(arr1[i], arr2[i]);
    }
}

void _double_pairwise_min(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v;
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_min_vec(_double_loadu(&arr1[i]), _double_loadu(&arr2[i]));
        _double_storeu(&outarr[i], v);
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = min(arr1[i], arr2[i]);
    }
}

void _float_pairwise_min(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v;
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_min_vec(_float_loadu(&arr1[i]), _float_loadu(&arr2[i]));
        _float_storeu(&outarr[i], v);
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = min(arr1[i], arr2[i]);
    }
}

void _double_running_sum(const double* arr, int len, int window, double* outarr) {
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

void _float_running_sum(const float* arr, int len, int window, float* outarr) {
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

void _double_add_arr(const double* arr1, const double* arr2, int len, double* outarr) {
    __double_vector v;
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        v = _double_add_vec(_double_loadu(&arr1[i]), _double_loadu(&arr2[i]));
        _double_storeu(&outarr[i], v);
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]+arr2[i];
    }
}

void _float_add_arr(const float* arr1, const float* arr2, int len, float* outarr) {
    __float_vector v;
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        v = _float_add_vec(_float_loadu(&arr1[i]), _float_loadu(&arr2[i]));
        _float_storeu(&outarr[i], v);
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]+arr2[i];
    }
}

void _double_memcpy(const double* src, int len, double* outarr) {
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i], _double_loadu(&src[i]));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = src[i];
    }
}

void _float_memcpy(const float* src, int len, float* outarr) {
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i], _float_loadu(&src[i]));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = src[i];
    }
}

void _double_div_diff(const double* arr1, const double* arr2, const double* arr3, int len, double* outarr) {
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i], _double_div_vec(_double_loadu(&arr1[i]),
                _double_sub_vec(_double_loadu(&arr2[i]),
                        _double_loadu(&arr3[i]))));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]/(arr2[i]-arr3[i]);
    }
}

void _float_div_diff(const float* arr1, const float* arr2, const float* arr3, int len, float* outarr) {
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i], _float_div_vec(_float_loadu(&arr1[i]),
                                                   _float_sub_vec(_float_loadu(&arr2[i]),
                                                                 _float_loadu(&arr3[i]))));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]/(arr2[i]-arr3[i]);
    }
}

void _double_mul_arr(const double* arr1, const double* arr2, int len, double* outarr) {
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                _double_mul_vec(_double_loadu(&arr1[i]),
                        _double_loadu(&arr2[i])));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]*arr2[i];
    }
}

void _float_mul_arr(const float* arr1, const float* arr2, int len, float* outarr) {
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                         _float_mul_vec(_float_loadu(&arr1[i]),
                                       _float_loadu(&arr2[i])));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = arr1[i]*arr2[i];
    }
}

void _double_cumsum(const double* arr1, int len, double* outarr) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr1[i];
        outarr[i] = sum;
    }
}

void _float_cumsum(const float* arr1, int len, float* outarr) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr1[i];
        outarr[i] = sum;
    }
}

void _double_running_sum_div(const double* arr1, const double* arr2, int len, int window, double* outarr) {
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

void _float_running_sum_div(const float* arr1, const float* arr2, int len, int window, float* outarr) {
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

void _double_vec_mean(const double** arrm, int n, int len, double* outarrm) {
    __double_vector vsum;
    double sum;
    double sf = 1./n;
    __double_vector s = _double_set1_vec(sf);

    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        vsum = _double_setzero_vec();
        for (int j = 0; j < n; j++) {
            vsum = _double_add_vec(vsum, _double_loadu(&arrm[j][i]));
        }
        vsum = _double_mul_vec(vsum, s);
        _double_storeu(&outarrm[i], vsum);
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        sum = 0;
        for (int j = 0; j < n; j++) {
            sum += arrm[j][i];
        }
        outarrm[i] = sum*sf;
    }
}

void _float_vec_mean(const float** arrm, int n, int len, float* outarrm) {
    __float_vector vsum;
    float sum;
    float sf = 1.f/n;
    __float_vector s = _float_set1_vec(sf);

    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        vsum = _float_setzero_vec();
        for (int j = 0; j < n; j++) {
            vsum = _float_add_vec(vsum, _float_loadu(&arrm[j][i]));
        }
        vsum = _float_mul_vec(vsum, s);
        _float_storeu(&outarrm[i], vsum);
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        sum = 0;
        for (int j = 0; j < n; j++) {
            sum += arrm[j][i];
        }
        outarrm[i] = sum*sf;
    }
}

void _double_mfi_algo(const double* high, const double* low, const double* close, const double* volume, int n, int len, double* outarr) {
    double psum = 0;
    double* pvals = malloc(n*sizeof(double));
    double nsum = 0;
    double* nvals = malloc(n*sizeof(double));
    double t;
    double lt;
    int index = 0;

    const double* arrm[3] = {high+n, low+n, close+n};
    _double_vec_mean(arrm, 3, len-n, outarr);

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

void _float_mfi_algo(const float* high, const float* low, const float* close, const float* volume, int n, int len, float* outarr) {
    float psum = 0;
    float* pvals = malloc(n*sizeof(float));
    float nsum = 0;
    float* nvals = malloc(n*sizeof(float));
    float t;
    float lt;
    int index = 0;

    const float* arrm[3] = {high+n, low+n, close+n};
    _float_vec_mean(arrm, 3, len-n, outarr);

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

void _double_recip(const double* arr1, int len, double* outarr) {
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                       _double_recp_vec(_double_loadu(&arr1[i])));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = 1./arr1[i];
    }
}

void _float_recip(const float* arr1, int len, float* outarr) {
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                       _float_recp_vec(_float_loadu(&arr1[i])));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = 1.f/arr1[i];
    }
}

void _double_nvi(const double* close, const double* volume, int len, double* outarr) {
    outarr[0] = 1000.;
    for (int i = 1; i < len; i++) {
        if (volume[i] < volume[i-1]) {
            outarr[i] = outarr[i-1]*close[i]/close[i-1];
        } else {
            outarr[i] = outarr[i-1];
        }
    }
}

void _float_nvi(const float* close, const float* volume, int len, float* outarr) {
    outarr[0] = 1000.f;
    for (int i = 1; i < len; i++) {
        if (volume[i] < volume[i-1]) {
            outarr[i] = outarr[i-1]*close[i]/close[i-1];
        } else {
            outarr[i] = outarr[i-1];
        }
    }
}

void _double_obv(const double* close, const double* volume, int len, double* outarr) {
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
void _float_obv(const float* close, const float* volume, int len, float* outarr) {
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

void _double_vpt(const double* close, const double* volume, int len, double* outarr) {
    outarr[0] = 0;
    for (int i = 1; i < len; i++) {
        outarr[i] = outarr[i-1] + volume[i]*(close[i]/close[i-1]-1);
    }
}

void _float_vpt(const float* close, const float* volume, int len, float* outarr) {
    outarr[0] = 0;
    for (int i = 1; i < len; i++) {
        outarr[i] = outarr[i - 1] + volume[i] * (close[i] / close[i - 1] - 1);
    }
}

void _double_vwap(const double* high, const double* low, const double* close,
                  const double* volume, int n, int len, double* outarr) {
    double s1 = 0;
    double s2 = 0;
    double* ps1 = malloc(n*sizeof(double));
    double* ps2 = malloc(n*sizeof(double));
    int index = 0;

    const double* arrm[3] = {high, low, close};
    _double_vec_mean(arrm, 3, len, outarr);
    _double_mul_arr(outarr, volume, len, outarr);

    s1 = _double_total_sum(volume, n);
    s2 = _double_total_sum(outarr, n);
    _double_memcpy(volume, n, ps1);
    _double_memcpy(outarr, n, ps2);

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

    _double_set_nan(outarr, n);

    free(ps1);
    free(ps2);
}

void _float_vwap(const float* high, const float* low, const float* close,
                 const float* volume, int n, int len, float* outarr) {
    float s1 = 0;
    float s2 = 0;
    float* ps1 = malloc(n*sizeof(float));
    float* ps2 = malloc(n*sizeof(float));
    int index = 0;

    const float* arrm[3] = {high, low, close};
    _float_vec_mean(arrm, 3, len, outarr);
    _float_mul_arr(outarr, volume, len, outarr);

    s1 = _float_total_sum(volume, n);
    s2 = _float_total_sum(outarr, n);
    _float_memcpy(volume, n, ps1);
    _float_memcpy(outarr, n, ps2);

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

    _float_set_nan(outarr, n);

    free(ps1);
    free(ps2);
}

void _double_tr(const double* high, const double* low, const double* close,
                int len, double* outarr) {
    __double_vector v1;
    __double_vector v2;
    __double_vector v3;
    double n1;
    double n2;
    double n3;
    const __double_vector sign_mask = _double_set1_vec(-0.);

    outarr[0] = high[0]-low[0];

    for (int i = 1; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
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

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
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



void _float_tr(const float* high, const float* low, const float* close,
               int len, float* outarr) {
    __float_vector v1;
    __float_vector v2;
    __float_vector v3;
    float n1;
    float n2;
    float n3;
    const __float_vector sign_mask = _float_set1_vec(-0.);

    outarr[0] = high[0]-low[0];

    for (int i = 1; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
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

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
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


void _double_rsqrt(const double* arr1, int len, double* outarr) {
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                       _double_rsqrt_vec(_double_loadu(&arr1[i])));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = 1./sqrt(arr1[i]);
    }
}

void _float_rsqrt(const float* arr1, int len, float* outarr) {
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                       _float_rsqrt_vec(_float_loadu(&arr1[i])));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = 1.f/sqrtf(arr1[i]);
    }
}

void _double_sqrt(const double* arr1, int len, double* outarr) {
    for (int i = 0; i < len - len % DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_storeu(&outarr[i],
                       _double_sqrt_vec(_double_loadu(&arr1[i])));
    }

    for (int i = len - len % DOUBLE_VEC_SIZE; i < len; i++) {
        outarr[i] = sqrt(arr1[i]);
    }
}

void _float_sqrt(const float* arr1, int len, float* outarr) {
    for (int i = 0; i < len - len % FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_storeu(&outarr[i],
                      _float_sqrt_vec(_float_loadu(&arr1[i])));
    }

    for (int i = len - len % FLOAT_VEC_SIZE; i < len; i++) {
        outarr[i] = sqrtf(arr1[i]);
    }
}

void _double_running_stddev(const double* arr1, const double* arr2, int len, int n, double* outarr) {
    double* temp1 = malloc(n*sizeof(double));

    for (int i = 0; i < n; i++) {
        _double_sub(arr1, i+1, arr2[i], temp1);
        _double_square(temp1, i+1, temp1);
        outarr[i] = _double_total_sum(temp1, i+1)/(i+1);
    }

    for (int i = n; i < len; i++) {
        _double_sub(arr1+i-n+1, n, arr2[i], temp1);
        _double_square(temp1, n, temp1);
        outarr[i] = _double_total_sum(temp1, n);
    }

    _double_div(outarr+n, len-n, n, outarr+n);
    _double_sqrt(outarr, len, outarr);
    free(temp1);
}

void _float_running_stddev(const float* arr1, const float* arr2, int len, int n, float* outarr) {
    float* temp1 = malloc(n*sizeof(float));

    for (int i = 0; i < n; i++) {
        _float_sub(arr1, i+1, arr2[i], temp1);
        _float_square(temp1, i+1, temp1);
        outarr[i] = _float_total_sum(temp1, i+1)/(i+1);
    }

    for (int i = n; i < len; i++) {
        _float_sub(arr1+i-n+1, n, arr2[i], temp1);
        _float_square(temp1, n, temp1);
        outarr[i] = _float_total_sum(temp1, n);
    }

    _float_div(outarr+n, len-n, n, outarr+n);
    _float_sqrt(outarr, len, outarr);
    free(temp1);
}