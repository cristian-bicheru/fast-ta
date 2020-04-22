#include <stdlib.h>
#include <stdio.h>

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