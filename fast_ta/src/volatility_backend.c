#include <stdlib.h>
#include <stdio.h>

#include "2darray.h"
#include "funcs.h"

double* _ATR_DOUBLE(const double* high, const double* low, const double* close,
                    int len, int n) {
    double* atr = malloc(len*sizeof(double));
    _double_tr(high, low, close, len, atr);
    _double_ema(atr, len, 1./(double) n, atr);
    return atr;
}

float* _ATR_FLOAT(const float* high, const float* low, const float* close,
                  int len, int n) {
    float* atr = malloc(len*sizeof(float));
    _float_tr(high, low, close, len, atr);
    _float_ema(atr, len, 1.f/(float) n, atr);
    return atr;
}

double** _BOL_DOUBLE(const double* close, int len, int n, float ndev) {
    double** ret = double_malloc2d(3, len);
    _double_sma(close, len, n, ret[1]);
    _double_running_stddev(close, ret[1], len, n, ret[0]);
    _double_mul(ret[0], len, (double) ndev, ret[0]);
    _double_memcpy(ret[1], len, ret[2]);
    _double_sub_arr(ret[2], ret[0], len, ret[2]);
    _double_add_arr(ret[1], ret[0], len, ret[0]);
    return ret;
}

float** _BOL_FLOAT(const float* close, int len, int n, float ndev) {
    float** ret = float_malloc2d(3, len);
    _float_sma(close, len, n, ret[1]);
    _float_running_stddev(close, ret[1], len, n, ret[0]);
    _float_mul(ret[0], len, (float) ndev, ret[0]);
    _float_memcpy(ret[1], len, ret[2]);
    _float_sub_arr(ret[2], ret[0], len, ret[2]);
    _float_add_arr(ret[1], ret[0], len, ret[0]);
    return ret;
}