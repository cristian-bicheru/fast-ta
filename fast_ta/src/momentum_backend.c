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

double* _RSI_DOUBLE(const double* close, double* out, int close_len, int _n,
                    int prelim) {
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
        for (int i = 0; i<_n; i++) {
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

    int start = prelim == 0 ? _n+1 : 1;
    for (int i = start; i < close_len; i++) {
        double diff = close[i]-close[i-1];
        rsi[i] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, _n,
                                      double);
    }

    if (prelim == 0) {
        rsi[0] = rsi[1];
    }

    return rsi;
}

float* _RSI_FLOAT(const float* close, float* out, int close_len, int _n,
                  int prelim) {
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
        for (int i = 0; i<_n; i++) {
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

    int start = prelim == 0 ? _n+1 : 1;
    for (int i = start; i < close_len; i++) {
        float diff = close[i]-close[i-1];
        rsi[i] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, _n,
                                      float);
    }

    if (prelim == 0) {
        rsi[0] = rsi[1];
    }

    return rsi;
}

double* _AO_DOUBLE(const double* high, const double*  low, int n1, int n2, int len) {
    double* median = malloc(len*sizeof(double));
    double* sma1 = malloc(len*sizeof(double));
    double* sma2 = malloc(len*sizeof(double));
    _double_pairwise_mean(high, low, len, median);
    _double_sma(median, len, n1, sma1);
    _double_sma(median, len, n2, sma2);
    _double_sub(sma1, sma2, median, len);
    free(sma1);
    free(sma2);
    return median;
}

float* _AO_FLOAT(const float* high, const float* low, int n1, int n2, int len) {
    float* median = malloc(len*sizeof(float));
    float* sma1 = malloc(len*sizeof(float));
    float* sma2 = malloc(len*sizeof(float));
    _float_pairwise_mean(high, low, len, median);
    _float_sma(median, len, n1, sma1);
    _float_sma(median, len, n2, sma2);
    _float_sub(sma1, sma2, median, len);
    free(sma1);
    free(sma2);
    return median;
}

double* _KAMA_DOUBLE(const double* close, int n1, int n2, int n3, int len) {
    double* change = malloc((len-n1)*sizeof(double));
    double* vol_sum = malloc(len*sizeof(double));
    double* sc = malloc(len*sizeof(double));
    _double_sub(close+n1, close, change, len-n1);
    _double_abs(change, len-n1, change);
    _double_volatility_sum(close, n1, len, vol_sum);
    _double_div_arr(change, vol_sum, len-n1, sc+n1);
    _double_mul(sc+n1, len-n1, 2./(n2+1)-2./(n3+1), sc+n1);
    _double_add(sc+n1, len-n1, 2./(n3+1), sc+n1);
    _double_square(sc+n1, len-n1, sc+n1);
    sc[n1] = close[n1];

    for (int i = 1; i < len-n1; i++) {
        sc[i+n1] = sc[i+n1-1]+sc[i+n1]*(close[i+n1] - sc[i+n1-1]);
    }

    _double_set_nan(sc, n1);

    free(change);
    free(vol_sum);
    return sc;
}

float* _KAMA_FLOAT(const float* close, int n1, int n2, int n3, int len) {
    float* change = malloc((len-n1)*sizeof(float));
    float* vol_sum = malloc(len*sizeof(float));
    float* sc = malloc(len*sizeof(float));
    _float_sub(close+n1, close, change, len-n1);
    _float_abs(change, len-n1, change);
    _float_volatility_sum(close, n1, len, vol_sum);
    _float_div_arr(change, vol_sum, len-n1, sc+n1);
    _float_mul(sc+n1, len-n1, 2./(n2+1)-2./(n3+1), sc+n1);
    _float_add(sc+n1, len-n1, 2./(n3+1), sc+n1);
    _float_square(sc+n1, len-n1, sc+n1);
    sc[n1] = close[n1];

    for (int i = 1; i < len-n1; i++) {
        sc[i+n1] = sc[i+n1-1]+sc[i+n1]*(close[i+n1] - sc[i+n1-1]);
    }

    _float_set_nan(sc, n1);

    free(change);
    free(vol_sum);
    return sc;
}

double* _ROC_DOUBLE(const double* close, int n, int len) {
    double* roc = malloc(len*sizeof(double));
    _double_sub(close+n, close, roc+n, len-n);
    _double_div_arr(roc+n, close, len-n, roc+n);
    _double_mul(roc+n, len-n, 100., roc+n);
    _double_set_nan(roc, n);
    return roc;
}

float* _ROC_FLOAT(const float* close, int n, int len) {
    float* roc = malloc(len*sizeof(float));
    _float_sub(close+n, close, roc+n, len-n);
    _float_div_arr(roc+n, close, len-n, roc+n);
    _float_mul(roc+n, len-n, 100.f, roc+n);
    _float_set_nan(roc, n);
    return roc;
}

struct double_array_pair _STOCHASTIC_OSCILLATOR_DOUBLE(const double* high, const double* low, double* close, int n, int d, int len) {
    // NOTE: While running_max and running_min are initially running maxes and minimums,
    //       they are reused later on save memory.
    double* running_max = malloc((len-n)*sizeof(double));
    double* running_min = malloc((len-n)*sizeof(double));

    _double_running_max(high, len, n, running_max);
    _double_running_min(low, len, n, running_min);
    _double_sub(running_max, running_min, running_max, len-n);
    _double_sub(close+n, running_min, running_min, len-n);

    _double_div_arr(running_min, running_max, len-n, running_min);
    _double_mul(running_min, len-n, 100., running_min);
    _double_sma(running_min, len-n, d, running_max);

    struct double_array_pair ret = {running_min, running_max};
    return ret;
}

struct float_array_pair _STOCHASTIC_OSCILLATOR_FLOAT(const float* high, const float* low, float* close, int n, int d, int len) {
    // NOTE: While running_max and running_min are initially running maxes and minimums,
    //       they are reused later on save memory.
    float* running_max = malloc((len-n)*sizeof(float));
    float* running_min = malloc((len-n)*sizeof(float));

    _float_running_max(high, len, n, running_max);
    _float_running_min(low, len, n, running_min);
    _float_sub(running_max, running_min, running_max, len-n);
    _float_sub(close+n, running_min, running_min, len-n);

    _float_div_arr(running_min, running_max, len-n, running_min);
    _float_mul(running_min, len-n, 100.f, running_min);
    _float_sma(running_min, len-n, d, running_max);

    struct float_array_pair ret = {running_min, running_max};
    return ret;
}
