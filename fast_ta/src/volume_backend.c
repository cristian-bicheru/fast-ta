#include <stdlib.h>
#include <stdio.h>

#include "array_pair.h"
#include "funcs.h"

double* _ADI_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len) {
    double* adi = malloc(len*sizeof(double));

    _double_memcpy(close, len, adi);
    _double_mul(adi, len, 2., adi);
    _double_sub_arr(adi, low, len, adi);
    _double_sub_arr(adi, high, len, adi);
    _double_div_diff(adi, high, low, len, adi);
    _double_mul_arr(adi, volume, len, adi);
    _double_cumsum(adi, len, adi);

    return adi;
}


float* _ADI_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len) {
    float* adi = malloc(len*sizeof(float));

    _float_memcpy(close, len, adi);
    _float_mul(adi, len, 2.f, adi);
    _float_sub_arr(adi, low, len, adi);
    _float_sub_arr(adi, high, len, adi);
    _float_div_diff(adi, high, low, len, adi);
    _float_mul_arr(adi, volume, len, adi);
    _float_cumsum(adi, len, adi);

    return adi;
}

double* _CMF_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len, int n) {
    double* cmf = malloc(len*sizeof(double));

    _double_memcpy(close, len, cmf);
    _double_mul(cmf, len, 2.f, cmf);
    _double_sub_arr(cmf, low, len, cmf);
    _double_sub_arr(cmf, high, len, cmf);
    _double_div_diff(cmf, high, low, len, cmf);
    _double_mul_arr(cmf, volume, len, cmf);
    _double_running_sum_div(cmf, volume, len, n, cmf);

    return cmf;
}
float* _CMF_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len, int n) {
    float* cmf = malloc(len*sizeof(float));

    _float_memcpy(close, len, cmf);
    _float_mul(cmf, len, 2.f, cmf);
    _float_sub_arr(cmf, low, len, cmf);
    _float_sub_arr(cmf, high, len, cmf);
    _float_div_diff(cmf, high, low, len, cmf);
    _float_mul_arr(cmf, volume, len, cmf);
    _float_running_sum_div(cmf, volume, len, n, cmf);

    return cmf;
}

struct double_array_pair _EMV_DOUBLE(const double* high, const double* low,
                                     const double* volume, int len, int n) {
    double* temp1 = malloc(len*sizeof(double));
    double* temp2 = malloc(len*sizeof(double));

    _double_pairwise_mean(high, low, len, temp1);
    _double_sub_arr(temp1+1, temp1, len-1, temp2+1);
    _double_sub_arr(high, low, len, temp1);
    _double_mul_arr(temp2, temp1, len, temp1);
    _double_div_arr(temp1, volume, len, temp1);
    _double_mul(temp1, len, 100000000., temp1);

    _double_sma(temp1+1, len-1, n,temp2+1);

    _double_set_nan(temp1, 1);
    _double_set_nan(temp2, 1);

    struct double_array_pair ret = {temp1, temp2};
    return ret;
}

struct float_array_pair _EMV_FLOAT(const float* high, const float* low,
                                   const float* volume, int len, int n) {
    float* temp1 = malloc(len*sizeof(float));
    float* temp2 = malloc(len*sizeof(float));

    _float_pairwise_mean(high, low, len, temp1);
    _float_sub_arr(temp1+1, temp1, len-1, temp2+1);
    _float_sub_arr(high, low, len, temp1);
    _float_mul_arr(temp2, temp1, len, temp1);
    _float_div_arr(temp1, volume, len, temp1);
    _float_mul(temp1, len, 100000000., temp1);

    _float_sma(temp1+1, len-1, n,temp2+1);

    _float_set_nan(temp1, 1);
    _float_set_nan(temp2, 1);

    struct float_array_pair ret = {temp1, temp2};
    return ret;
}