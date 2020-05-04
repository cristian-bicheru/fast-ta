#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>

#include "funcs.h"
#include "momentum_backend.h"

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
            rsi[i+1] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, window_size,
                                            double);
        }
    } else {
        // some funky pointer math to move the close back the specified amount
        // so we can get the moving averages to where they should be.
        // NOTE: This is the dangerous part of the code.
        const double* reduced_close = close - prelim;
        for (int i = 0; i<prelim + 1; i++) {
            double diff = reduced_close[i+1]-reduced_close[i];
            COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, window_size, double);
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
            rsi[i+1] = COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, window_size,
                                            float);
        }
    } else {
        // some funky pointer math to move the close back the specified amount
        // so we can get the moving averages to where they should be.
        // NOTE: This is the dangerous part of the code.
        const float* reduced_close = close - prelim;
        for (int i = 0; i<prelim + 1; i++) {
            float diff = reduced_close[i+1]-reduced_close[i];
            COMPUTE_RSI_TEMPLATE(diff, &last_gain, &last_loss, window_size, float);
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

double* _AO_DOUBLE(const double* high, const double*  low, int n1, int n2, int len) {
    double* median = malloc(len*sizeof(double));
    double* sma1 = malloc(len*sizeof(double));
    double* sma2 = malloc(len*sizeof(double));
    _double_pairwise_mean(high, low, len, median);
    _double_sma(median, len, n1, sma1);
    _double_sma(median, len, n2, sma2);
    _double_sub_arr(sma1, sma2, len, median);
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
    _float_sub_arr(sma1, sma2, len, median);
    free(sma1);
    free(sma2);
    return median;
}

double* _KAMA_DOUBLE(const double* close, int n1, int n2, int n3, int len) {
    double* kama = malloc(len*sizeof(double));

    _double_er(close, len, n1, kama);
    _double_mul(kama, len, 2./(n2+1)-2./(n3+1), kama);
    _double_add(kama, len, 2./(n3+1), kama);
    _double_square(kama, len, kama);
    kama[0] = close[0];

    for (int i = 1; i < len; i++) {
        kama[i] = kama[i-1]+kama[i]*(close[i] - kama[i-1]);
    }

    return kama;
}

float* _KAMA_FLOAT(const float* close, int n1, int n2, int n3, int len) {
    float* kama = malloc(len*sizeof(float));

    _float_er(close, len, n1, kama);
    _float_mul(kama, len, 2.f/(n2+1)-2./(n3+1), kama);
    _float_add(kama, len, 2.f/(n3+1), kama);
    _float_square(kama, len, kama);
    kama[0] = close[0];

    for (int i = 1; i < len; i++) {
        kama[i] = kama[i-1]+kama[i]*(close[i] - kama[i-1]);
    }

    return kama;
}

double* _ROC_DOUBLE(const double* close, int n, int len) {
    double* roc = malloc( len*sizeof(double));
    _intrin_fast_double_roc(close, roc, len, n);
    _double_set_nan(roc, n);
    return roc;
}

float* _ROC_FLOAT(const float* close, int n, int len) {
    float* roc = malloc( len*sizeof(float));
    _float_sub_arr(close + n, close, len - n, roc + n);
    _float_div_arr(roc+n, close, len-n, roc+n);
    _float_mul(roc+n, len-n, 100.f, roc+n);
    _float_set_nan(roc, n);
    return roc;
}

double* _STOCHASTIC_OSCILLATOR_DOUBLE(const double* high, const double* low,
                                       const double* close, int n, int d,
                                       int len) {
    double* arr = malloc(2*len*sizeof(double));
    double* running_min = arr;
    double* running_max = arr+len;

    _double_running_max(high, len, n, running_max);
    _double_running_min(low, len, n, running_min);

    _double_sub_arr(running_max, running_min, len,
                        running_max);
    _double_sub_arr(close, running_min,
                        len, running_min);
    _double_div_arr(running_min, running_max, len, running_min);
    _double_mul(running_min, len, 100., running_min);
    _double_sma(running_min, len, d, running_max);

    return arr;
}

float *_STOCHASTIC_OSCILLATOR_FLOAT(const float *high, const float *low,
                                    const float *close, int n, int d, int len) {
    float* arr = malloc(2*len*sizeof(float));
    float* running_min = arr;
    float* running_max = arr+len;

    _float_running_max(high, len, n, running_max);
    _float_running_min(low, len, n, running_min);

    _float_sub_arr(running_max, running_min, len,
                    running_max);
    _float_sub_arr(close, running_min,
                    len, running_min);
    _float_div_arr(running_min, running_max, len, running_min);
    _float_mul(running_min, len, 100.f, running_min);
    _float_sma(running_min, len, d, running_max);

    return arr;
}

double* _TSI_DOUBLE(const double* close, int r, int s, int len) {
    double* pc = malloc(len*sizeof(double));
    double* apc = malloc(len*sizeof(double));
    _double_consecutive_diff(close, len, pc);
    _double_abs(pc, len, apc);
    _double_tsi_fast_ema(pc+1, apc+1, len-1, r, s);

    _double_div_arr(pc, apc, len, pc);
    _double_mul(pc, len, 100., pc);


    _double_set_nan(pc, 1);

    free(apc);
    return pc;
}

float* _TSI_FLOAT(const float* close, int r, int s, int len) {
    float* pc = malloc(len*sizeof(float));
    float* apc = malloc(len*sizeof(float));
    _float_consecutive_diff(close, len, pc);
    _float_abs(pc, len, apc);
    _float_tsi_fast_ema(pc+1, apc+1, len-1, r, s);

    _float_div_arr(pc, apc, len, pc);
    _float_mul(pc, len, 100.f, pc);


    _float_set_nan(pc, 1);

    free(apc);
    return pc;
}

double* _ULTIMATE_OSCILLATOR_DOUBLE(const double* high, const double* low, const double* close, int s, int m, int l, double ws, double wm, double wl, int len) {
    int nan_range = max(max(s, m), l);
    double* bp = malloc((len-1)*sizeof(double));
    double* tr = malloc((len-1)*sizeof(double));
    double* avgbp = malloc(len*sizeof(double));
    double* avgtr = malloc(len*sizeof(double));
    double* uo = malloc(len*sizeof(double));

    _double_pairwise_min(low+1, close, len-1, bp);
    _double_pairwise_max(high+1, close, len-1, tr);
    _double_sub_arr(tr, bp, len - 1, tr);
    _double_sub_arr(close+1, bp, len-1, bp);

    _double_running_sum(bp, len-1, s, avgbp);
    _double_running_sum(tr, len-1, s, avgtr);
    _double_div_arr(avgbp, avgtr, len-1, uo+1);
    _double_mul(uo+1, len-1, ws, uo+1);

    _double_running_sum(bp, len-1, m, avgbp);
    _double_running_sum(tr, len-1, m, avgtr);
    _double_div_arr(avgbp, avgtr, len, avgbp);
    _double_mul(avgbp, len, wm, avgbp);
    _double_add_arr(avgbp, uo+1, len-1, uo+1);

    _double_running_sum(bp, len-1, l, avgbp);
    _double_running_sum(tr, len-1, l, avgtr);
    _double_div_arr(avgbp, avgtr, len, avgbp);
    _double_mul(avgbp, len, wl, avgbp);
    _double_add_arr(avgbp, uo+1, len-1, uo+1);

    _double_mul(uo+1, len-1, 100./(ws+wm+wl), uo+1);

    _double_set_nan(uo, nan_range);

    free(bp);
    free(tr);
    free(avgbp);
    free(avgtr);

    return uo;
}

float* _ULTIMATE_OSCILLATOR_FLOAT(const float* high, const float* low, const float* close, int s, int m, int l, double ws, double wm, double wl, int len) {
    int nan_range = max(max(s, m), l);
    float* bp = malloc((len-1)*sizeof(float));
    float* tr = malloc((len-1)*sizeof(float));
    float* avgbp = malloc(len*sizeof(float));
    float* avgtr = malloc(len*sizeof(float));
    float* uo = malloc(len*sizeof(float));

    _float_pairwise_min(low+1, close, len-1, bp);
    _float_pairwise_max(high+1, close, len-1, tr);
    _float_sub_arr(tr, bp, len - 1, tr);
    _float_sub_arr(close+1, bp, len-1, bp);

    _float_running_sum(bp, len-1, s, avgbp);
    _float_running_sum(tr, len-1, s, avgtr);
    _float_div_arr(avgbp, avgtr, len-1, uo+1);
    _float_mul(uo+1, len-1, ws, uo+1);

    _float_running_sum(bp, len-1, m, avgbp);
    _float_running_sum(tr, len-1, m, avgtr);
    _float_div_arr(avgbp, avgtr, len, avgbp);
    _float_mul(avgbp, len, wm, avgbp);
    _float_add_arr(avgbp, uo+1, len-1, uo+1);

    _float_running_sum(bp, len-1, l, avgbp);
    _float_running_sum(tr, len-1, l, avgtr);
    _float_div_arr(avgbp, avgtr, len, avgbp);
    _float_mul(avgbp, len, wl, avgbp);
    _float_add_arr(avgbp, uo+1, len-1, uo+1);

    _float_mul(uo+1, len-1, 100.f/(ws+wm+wl), uo+1);

    _float_set_nan(uo, nan_range);

    free(bp);
    free(tr);
    free(avgbp);
    free(avgtr);

    return uo;
}

double* _WILLIAMS_R_DOUBLE(const double* high, const double* low, const double* close, int n,
                           int len) {
    double* running_min = malloc(len*sizeof(double));
    double* running_max = malloc(len*sizeof(double));

    _double_running_max(high, len, n, running_max);
    _double_running_min(low, len, n, running_min);

    _double_sub_arr(running_max, running_min, len,
                        running_min);
    _double_sub_arr(running_max, close,
                        len, running_max);
    _double_div_arr(running_max, running_min, len, running_min);
    _double_mul(running_min, len, -100., running_min);

    free(running_max);

    return running_min;
}

float* _WILLIAMS_R_FLOAT(const float* high, const float* low, const float* close, int n,
                         int len) {
    float* running_min = malloc(len*sizeof(float));
    float* running_max = malloc(len*sizeof(float));

    _float_running_max(high, len, n, running_max);
    _float_running_min(low, len, n, running_min);

    _float_sub_arr(running_max, running_min, len,
                    running_min);
    _float_sub_arr(running_max, close,
                    len, running_max);
    _float_div_arr(running_max, running_min, len, running_min);
    _float_mul(running_min, len, -100.f, running_min);

    free(running_max);

    return running_min;
}