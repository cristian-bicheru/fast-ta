#include <stdlib.h>
#include <stdio.h>

#include "../funcs/funcs_unaligned.h"
#include "../2darray.h"
#include "../funcs/funcs.h"

double* _ATR_DOUBLE(const double* high, const double* low, const double* close,
                    int len, int n) {
    double* atr = double_malloc(len);
    _double_atr(high, low, close, len, n, atr);
    return atr;
}

float* _ATR_FLOAT(const float* high, const float* low, const float* close,
                  int len, int n) {
    float* atr = float_malloc(len);
    _float_atr(high, low, close, len, n, atr);
    return atr;
}

double** _BOL_DOUBLE(const double* close, int len, int n, float ndev) {
    double** ret = double_malloc2d(3, len);
    _double_sma(close, len, n, ret[1]);
    _double_running_stddev(close, ret[1], len, n, ret[0]);
    _double_mul(ret[0], len, (double) ndev, ret[0]);
    _double_memcpy_unaligned(ret[1], len, ret[2]);
    _double_sub_arr_unaligned(ret[2], ret[0], len, ret[2]);
    _double_add_arr_unaligned(ret[1], ret[0], len, ret[0]);
    return ret;
}

float** _BOL_FLOAT(const float* close, int len, int n, float ndev) {
    float** ret = float_malloc2d(3, len);
    _float_sma(close, len, n, ret[1]);
    _float_running_stddev(close, ret[1], len, n, ret[0]);
    _float_mul(ret[0], len, (float) ndev, ret[0]);
    _float_memcpy_unaligned(ret[1], len, ret[2]);
    _float_sub_arr_unaligned(ret[2], ret[0], len, ret[2]);
    _float_add_arr_unaligned(ret[1], ret[0], len, ret[0]);
    return ret;
}

double** _DC_DOUBLE(const double* high, const double* low, int len, int n) {
    double** dc = double_malloc2d(3, len);
    _double_running_max_unaligned(high, len, n, dc[0]);
    _double_running_min_unaligned(low, len, n, dc[2]);
    _double_pairwise_mean_unaligned(dc[0], dc[2], len, dc[1]);
    return dc;
}

float** _DC_FLOAT(const float* high, const float* low, int len, int n) {
    float** dc = float_malloc2d(3, len);
    _float_running_max_unaligned(high, len, n, dc[0]);
    _float_running_min_unaligned(low, len, n, dc[2]);
    _float_pairwise_mean_unaligned(dc[0], dc[2], len, dc[1]);
    return dc;
}

double** _KC_DOUBLE(const double* high, const double* low, const double* close,
                   int len, int n1, int n2, int num_channels) {
    double** dc = double_malloc2d(num_channels*2+1, len);
    const double* arrm[3] = {high, low, close};
    _double_vec_mean(arrm, 3, len, dc[num_channels]);
    _double_ema(dc[num_channels], len, 2./(n1+1.), dc[num_channels]);

    _double_atr(high, low, close, len, n2, dc[0]);
    for (int i = 0; i < num_channels; i++) {
        _double_add_arr_unaligned(dc[num_channels+i], dc[0], len, dc[num_channels+i+1]);
    }

    for (int i = 0; i < num_channels; i++) {
        _double_sub_arr_unaligned(dc[num_channels-i], dc[0], len, dc[num_channels-i-1]);
    }
    return dc;
}

float** _KC_FLOAT(const float* high, const float* low, const float* close,
                 int len, int n1, int n2, int num_channels) {
    float** dc = float_malloc2d(num_channels*2+1, len);
    const float* arrm[3] = {high, low, close};
    _float_vec_mean(arrm, 3, len, dc[num_channels]);
    _float_ema(dc[num_channels], len, 2.f/(n1+1.f), dc[num_channels]);

    _float_atr(high, low, close, len, n2, dc[0]);
    for (int i = 0; i < num_channels; i++) {
        _float_add_arr_unaligned(dc[num_channels+i], dc[0], len, dc[num_channels+i+1]);
    }

    for (int i = 0; i < num_channels; i++) {
        _float_sub_arr_unaligned(dc[num_channels-i], dc[0], len, dc[num_channels-i-1]);
    }
    return dc;
}