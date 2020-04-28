#include <stdlib.h>
#include <stdio.h>

#include "funcs.h"
#include "2darray.h"

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

double** _EMV_DOUBLE(const double* high, const double* low,
                                     const double* volume, int len, int n) {
    double** arr = double_malloc2d(2, len);
    double* temp1 = arr[0];
    double* temp2 = arr[1];

    _double_pairwise_mean(high, low, len, temp1);
    _double_sub_arr(temp1+1, temp1, len-1, temp2+1);
    _double_sub_arr(high, low, len, temp1);
    _double_mul_arr(temp2, temp1, len, temp1);
    _double_div_arr(temp1, volume, len, temp1);
    _double_mul(temp1, len, 100000000., temp1);

    _double_sma(temp1+1, len-1, n,temp2+1);

    _double_set_nan(temp1, 1);
    _double_set_nan(temp2, 1);

    return arr;
}

float** _EMV_FLOAT(const float* high, const float* low,
                                   const float* volume, int len, int n) {
    float** arr = float_malloc2d(2, len);
    float* temp1 = arr[0];
    float* temp2 = arr[1];

    _float_pairwise_mean(high, low, len, temp1);
    _float_sub_arr(temp1+1, temp1, len-1, temp2+1);
    _float_sub_arr(high, low, len, temp1);
    _float_mul_arr(temp2, temp1, len, temp1);
    _float_div_arr(temp1, volume, len, temp1);
    _float_mul(temp1, len, 100000000.f, temp1);

    _float_sma(temp1+1, len-1, n,temp2+1);

    _float_set_nan(temp1, 1);
    _float_set_nan(temp2, 1);

    return arr;
}

double* _FI_DOUBLE(const double* close, const double* volume, int len, int n) {
    double* fi = malloc(len*sizeof(double));
    _double_sub_arr(close+1, close, len-1, fi+1);
    _double_mul_arr(fi+1, volume+1, len-1, fi+1);
    _double_ema(fi+1, len-1, 2./(n+1.), fi+1);

    _double_set_nan(fi, 1);

    return fi;
}

float* _FI_FLOAT(const float* close, const float* volume, int len, int n) {
    float* fi = malloc(len*sizeof(float));
    _float_sub_arr(close+1, close, len-1, fi+1);
    _float_mul_arr(fi+1, volume+1, len-1, fi+1);
    _float_ema(fi+1, len-1, 2.f/(n+1.f), fi+1);

    _float_set_nan(fi, 1);

    return fi;
}

double* _MFI_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len, int n) {
    double* temp1 = malloc(len*sizeof(double));
    _double_mfi_algo(high, low, close, volume, n, len, temp1+n);
    _double_add(temp1+n, len-n, 1., temp1+n);
    _double_recip(temp1+n, len-n, temp1+n);
    _double_mul(temp1+n, len-n, -100., temp1+n);
    _double_add(temp1+n, len-n, 100., temp1+n);
    _double_set_nan(temp1, n);
    return temp1;
}

float* _MFI_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len, int n) {
    float* temp1 = malloc(len*sizeof(float));
    _float_mfi_algo(high, low, close, volume, n, len, temp1+n);
    _float_add(temp1+n, len-n, 1.f, temp1+n);
    _float_recip(temp1+n, len-n, temp1+n);
    _float_mul(temp1+n, len-n, -100.f, temp1+n);
    _float_add(temp1+n, len-n, 100.f, temp1+n);
    _float_set_nan(temp1, n);
    return temp1;
}

double* _NVI_DOUBLE(const double* close, const double* volume, int len) {
    double* nvi = malloc(len*sizeof(double));
    _double_nvi(close, volume, len, nvi);
    return nvi;
}

float* _NVI_FLOAT(const float* close, const float* volume, int len) {
    float* nvi = malloc(len*sizeof(float));
    _float_nvi(close, volume, len, nvi);
    return nvi;
}

double* _OBV_DOUBLE(const double* close, const double* volume, int len) {
    double* obv = malloc(len*sizeof(double));
    _double_obv(close, volume, len, obv);
    return obv;
}

float* _OBV_FLOAT(const float* close, const float* volume, int len) {
    float* obv = malloc(len*sizeof(float));
    _float_obv(close, volume, len, obv);
    return obv;
}

double* _VPT_DOUBLE(const double* close, const double* volume, int len) {
    double* vpt = malloc(len*sizeof(double));
    _double_vpt(close, volume, len, vpt);
    return vpt;
}

float* _VPT_FLOAT(const float* close, const float* volume, int len) {
    float* vpt = malloc(len*sizeof(float));
    _float_vpt(close, volume, len, vpt);
    return vpt;
}

double* _VWAP_DOUBLE(const double* high, const double* low, const double* close,
                     const double* volume, int len, int n) {
    double* vwap = malloc(len*sizeof(double));
    _double_vwap(high, low, close, volume, n, len, vwap);
    return vwap;
}
float* _VWAP_FLOAT(const float* high, const float* low, const float* close,
                   const float* volume, int len, int n) {
    float* vwap = malloc(len*sizeof(float));
    _float_vwap(high, low, close, volume, n, len, vwap);
    return vwap;

}