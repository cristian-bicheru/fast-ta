#include <stdlib.h>
#include <stdio.h>

#include "funcs.h"
#include "momentum_backend.h"
#include "array_pair.h"

#define COMPUTE_RSI_TEMPLATE(diff, last_gain, last_loss, window_size, TYPE) ({ \
    TYPE gain;                                                                 \
    TYPE loss;                                                                 \
    if ((diff) > 0) {                                                          \
        gain = (diff);                                                         \
        loss = 0;                                                              \
    } else if ((diff) < 0) {                                                   \
        gain = 0;                                                              \
        loss = -(diff);                                                        \
    } else {                                                                   \
        gain = 0;                                                              \
        loss = 0;                                                              \
    }                                                                          \
                                                                               \
    (*(last_gain)) =                                                           \
        ((*last_gain) * ((window_size) - 1) + gain) / (window_size);           \
    (*(last_loss)) =                                                           \
        ((*last_loss) * ((window_size) - 1) + loss) / (window_size);           \
                                                                               \
    TYPE rsi;                                                                  \
    if (*(last_loss) == 0) {                                                   \
        if (*(last_gain) == 0) {                                               \
            rsi = 50;                                                          \
        } else {                                                               \
            rsi = 100;                                                         \
        }                                                                      \
    } else {                                                                   \
        rsi = 100 * (                                                          \
                        1 -                                                    \
                        1/( 1 + (*(last_gain))/(*(last_loss)) )                \
                    );                                                         \
    }                                                                          \
                                                                               \
    rsi;                                                                       \
})

double* _RSI_DOUBLE(const double* close, double* out, int close_len,
                    int window_size, int prelim) {
    // support the user mallocing themselves OR this function allocating the memory
    double* rsi;
    if (out == NULL) {
        rsi = malloc(close_len * sizeof(double));
    } else {
        rsi = out;
    }

    double last_gain = 0;
    double last_loss = 0;
    // prelim == 0 is essentially just the normal behaviour of RSI
    if (prelim == 0) {
        for (int i = 0; i<window_size; i++) {
            double diff = close[i+1]-close[i];
            rsi[i+1] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, i+1,
                                            double);
        }
    } else {
        // some funky pointer math to move the close back the specified amount
        // so we can get the moving averages to where they should be.
        // NOTE: This is the dangerous part of the code.
        double* reduced_close = close - prelim;
        for (int i = 0; i<prelim + 1; i++) {
            double diff = reduced_close[i+1]-reduced_close[i];
            COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, i+1, double);
        }
    }

    int start = prelim == 0 ? window_size+1 : 1;
    for (int i = start; i < close_len; i++) {
        double diff = close[i]-close[i-1];
        rsi[i] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, window_size,
                                      double);
    }

    if (prelim == 0) {
        rsi[0] = rsi[1];
    }

    return rsi;
}

float* _RSI_FLOAT(const float* close, float* out, int close_len,
                  int window_size, int prelim) {
    // support the user mallocing themselves OR this function allocating the memory
    float* rsi;
    if (out == NULL) {
        rsi = malloc(close_len * sizeof(float));
    } else {
        rsi = out;
    }

    float last_gain = 0;
    float last_loss = 0;
    // prelim == 0 is essentially just the normal behaviour of RSI
    if (prelim == 0) {
        for (int i = 0; i<window_size; i++) {
            float diff = close[i+1]-close[i];
            rsi[i+1] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, i+1,
                                            float);
        }
    } else {
        // some funky pointer math to move the close back the specified amount
        // so we can get the moving averages to where they should be.
        // NOTE: This is the dangerous part of the code.
        float* reduced_close = close - prelim;
        for (int i = 0; i<prelim + 1; i++) {
            float diff = reduced_close[i+1]-reduced_close[i];
            COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, i+1, float);
        }
    }

    int start = prelim == 0 ? window_size+1 : 1;
    for (int i = start; i < close_len; i++) {
        float diff = close[i]-close[i-1];
        rsi[i] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, window_size,
                                      float);
    }

    if (prelim == 0) {
        rsi[0] = rsi[1];
    }

    return rsi;
}

double* _AO_DOUBLE(double* high, double*  low, int n1, int n2, int len) {
    double* median = _double_pairwise_mean(high, low, len);
    double* sma1 = _double_sma(median, len, n1);
    double* sma2 = _double_sma(median, len, n2);
    _double_sub(sma1, sma2, median, len);
    free(sma1);
    free(sma2);
    return median;
}

float* _AO_FLOAT(float* high, float* low, int n1, int n2, int len) {
    float* median = _float_pairwise_mean(high, low, len);
    float* sma1 = _float_sma(median, len, n1);
    float* sma2 = _float_sma(median, len, n2);
    _float_sub(sma1, sma2, median, len);
    free(sma1);
    free(sma2);
    return median;
}

double* _KAMA_DOUBLE(double* close, int n1, int n2, int n3, int len) {
    double* change = malloc((len-n1)*sizeof(double));
    _double_sub(close+n1, close, change, len-n1);
    _double_inplace_abs(change, len-n1);
    double* vol_sum = _double_volatility_sum(close, n1, len);
    double* sc = _double_div_arr(change, vol_sum, len-n1);
    _double_inplace_mul(sc, len-n1, 2./(n2+1)-2./(n3+1));
    _double_inplace_add(sc, len-n1, 2./(n3+1));
    _double_inplace_square(sc, len-n1);
    sc[0] = close[n1];

    for (int i = 1; i < len-n1; i++) {
        sc[i] = sc[i-1]+sc[i]*(close[i+n1] - sc[i-1]);
    }
    free(change);
    free(vol_sum);
    return sc;
}

float* _KAMA_FLOAT(float* close, int n1, int n2, int n3, int len) {
    float* change = malloc((len-n1)*sizeof(float));
    _float_sub(close+n1, close, change, len-n1);
    _float_inplace_abs(change, len-n1);
    float* vol_sum = _float_volatility_sum(close, n1, len);
    float* sc = _float_div_arr(change, vol_sum, len-n1);
    _float_inplace_mul(sc, len-n1, 2./(n2+1)-2./(n3+1));
    _float_inplace_add(sc, len-n1, 2./(n3+1));
    _float_inplace_square(sc, len-n1);
    sc[0] = close[n1];

    for (int i = 1; i < len-n1; i++) {
        sc[i] = sc[i-1]+sc[i]*(close[i+n1] - sc[i-1]);
    }
    free(change);
    free(vol_sum);
    return sc;
}

double* _ROC_DOUBLE(double* close, int n, int len) {
    double* roc = malloc((len-n)*sizeof(double));
    _double_sub(close+n, close, roc, len-n);
    _double_inplace_div_arr(roc, len-n, close);
    _double_inplace_mul(roc, len-n, 100.);
    return roc;
}

float* _ROC_FLOAT(float* close, int n, int len) {
    float* roc = malloc((len-n)*sizeof(float));
    _float_sub(close+n, close, roc, len-n);
    _float_inplace_div_arr(roc, len-n, close);
    _float_inplace_mul(roc, len-n, 100.f);
    return roc;
}

struct double_array_pair _STOCHASTIC_OSCILLATOR_DOUBLE(double* high, double* low, double* close, int n, int d, int len) {
    _double_inplace_running_max(high, len, n);
    _double_inplace_running_min(low, len, n);
    _double_sub(close+n, low+n, close+n, len-n);
    _double_sub(high+n, low+n, high+n, len-n);
    _double_inplace_div_arr(close+n, len-n, high+n);
    _double_inplace_mul(close+n, len-n, 100.);

    struct double_array_pair ret = {close, _double_sma(close+n, len-n, d)};
    return ret;
}

struct float_array_pair _STOCHASTIC_OSCILLATOR_FLOAT(float* high, float* low, float* close, int n, int d, int len) {
    _float_inplace_running_max(high, len, n);
    _float_inplace_running_min(low, len, n);
    _float_sub(close+n, low+n, close+n, len-n);
    _float_sub(high+n, low+n, high+n, len-n);
    _float_inplace_div_arr(close+n, len-n, high+n);
    _float_inplace_mul(close+n, len-n, 100.f);

    struct float_array_pair ret = {close, _float_sma(close+n, len-n, d)};
    return ret;
}
