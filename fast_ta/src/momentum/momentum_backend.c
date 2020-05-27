#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>

#include "../funcs/funcs_unaligned.h"
#include "../funcs/funcs.h"
#include "momentum_backend.h"

double* _RSI_DOUBLE(const double* close, int close_len, int _n) {
    double last_gain, last_loss, cd, cgain, closs;
    double* rsi = double_malloc(close_len);

    last_gain = 0;
    last_loss = 0;

    for (int i = 0; i<_n; i++) {
        cd = close[i+1]-close[i];
        if (cd > 0) {
            cgain = cd;
            closs = 0;
        } else if (cd < 0) {
            cgain = 0;
            closs = -cd;
        } else {
            cgain = 0;
            closs = 0;
        }

        last_gain = (last_gain*(_n-1)+cgain)/_n;
        last_loss = (last_loss*(_n-1)+closs)/_n;

        if (last_loss == 0) {
            if (last_gain == 0) {
                rsi[i+1] = 50;
            } else {
                rsi[i+1] = 100;
            }
        } else {
            rsi[i+1] = 100*(1-1/(1+last_gain/last_loss));
        }
    }

    for (int i = _n+1; i < close_len; i++) {
        cd = close[i]-close[i-1];

        if (cd > 0) {
            cgain = cd;
            closs = 0;
        } else if (cd < 0) {
            cgain = 0;
            closs = -cd;
        } else {
            cgain = 0;
            closs = 0;
        }

        last_gain = (last_gain*(_n-1)+cgain)/_n;
        last_loss = (last_loss*(_n-1)+closs)/_n;

        if (last_loss == 0) {
            if (last_gain == 0) {
                rsi[i] = 50;
            } else {
                rsi[i] = 100;
            }
        } else {
            rsi[i] = 100*(1-1/(1+last_gain/last_loss));
        }
    }

    rsi[0] = rsi[1];
    return rsi;
}

float* _RSI_FLOAT(const float* close, int close_len, int _n) {
    float last_gain, last_loss, cd, cgain, closs;
    float* rsi = float_malloc(close_len);

    last_gain = 0;
    last_loss = 0;

    for (int i = 0; i<_n; i++) {
        cd = close[i+1]-close[i];
        if (cd > 0) {
            cgain = cd;
            closs = 0;
        } else if (cd < 0) {
            cgain = 0;
            closs = -cd;
        } else {
            cgain = 0;
            closs = 0;
        }

        last_gain = (last_gain*(_n-1)+cgain)/_n;
        last_loss = (last_loss*(_n-1)+closs)/_n;

        if (last_loss == 0) {
            if (last_gain == 0) {
                rsi[i+1] = 50;
            } else {
                rsi[i+1] = 100;
            }
        } else {
            rsi[i+1] = 100*(1-1/(1+last_gain/last_loss));
        }
    }

    for (int i = _n+1; i < close_len; i++) {
        cd = close[i]-close[i-1];

        if (cd > 0) {
            cgain = cd;
            closs = 0;
        } else if (cd < 0) {
            cgain = 0;
            closs = -cd;
        } else {
            cgain = 0;
            closs = 0;
        }

        last_gain = (last_gain*(_n-1)+cgain)/_n;
        last_loss = (last_loss*(_n-1)+closs)/_n;

        if (last_loss == 0) {
            if (last_gain == 0) {
                rsi[i] = 50;
            } else {
                rsi[i] = 100;
            }
        } else {
            rsi[i] = 100*(1-1/(1+last_gain/last_loss));
        }
    }

    rsi[0] = rsi[1];
    return rsi;
}

double* _AO_DOUBLE(const double* high, const double* low, int n1, int n2, int len) {
    double* median = double_malloc(len);
    double* sma1 = double_malloc(len);
    double* sma2 = double_malloc(len);
    _double_pairwise_mean(high, low, len, median);
    _double_sma(median, len, n1, sma1);
    _double_sma(median, len, n2, sma2);
    _double_sub_arr(sma1, sma2, len, median);
    free(sma1);
    free(sma2);
    return median;
}

float* _AO_FLOAT(const float* high, const float* low, int n1, int n2, int len) {
    float* median = float_malloc(len);
    float* sma1 = float_malloc(len);
    float* sma2 = float_malloc(len);
    _float_pairwise_mean(high, low, len, median);
    _float_sma(median, len, n1, sma1);
    _float_sma(median, len, n2, sma2);
    _float_sub_arr(sma1, sma2, len, median);
    free(sma1);
    free(sma2);
    return median;
}

double* _KAMA_DOUBLE(const double* close, int n1, int n2, int n3, int len) {
    double* kama = double_malloc(len);

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
    float* kama = float_malloc(len);

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
    double* roc = double_malloc(len);
    _intrin_fast_double_roc(close, roc, len, n);
    _double_set_nan(roc, n);
    return roc;
}

float* _ROC_FLOAT(const float* close, int n, int len) {
    float* roc = float_malloc(len);
    _intrin_fast_float_roc(close, roc, len, n);
    _float_set_nan(roc, n);
    return roc;
}

double* _STOCHASTIC_OSCILLATOR_DOUBLE(const double* high, const double* low,
                                       const double* close, int n, int d,
                                       int len) {
    double* arr = double_malloc(2*len);
    double* running_min = arr;
    double* running_max = arr+len;

    _double_running_max_unaligned(high, len, n, running_max);
    _double_running_min_unaligned(low, len, n, running_min);

    _double_sub_arr_unaligned(running_max, running_min, len,
                        running_max);
    _double_sub_arr_unaligned(close, running_min,
                        len, running_min);
    _double_div_arr_unaligned(running_min, running_max, len, running_min);
    _double_mul(running_min, len, 100., running_min);
    _double_sma(running_min, len, d, running_max);

    return arr;
}

float *_STOCHASTIC_OSCILLATOR_FLOAT(const float *high, const float *low,
                                    const float *close, int n, int d, int len) {
    float* arr = float_malloc(2*len);
    float* running_min = arr;
    float* running_max = arr+len;

    _float_running_max_unaligned(high, len, n, running_max);
    _float_running_min_unaligned(low, len, n, running_min);

    _float_sub_arr_unaligned(running_max, running_min, len,
                    running_max);
    _float_sub_arr_unaligned(close, running_min,
                    len, running_min);
    _float_div_arr_unaligned(running_min, running_max, len, running_min);
    _float_mul(running_min, len, 100.f, running_min);
    _float_sma(running_min, len, d, running_max);

    return arr;
}

double* _TSI_DOUBLE(const double* close, int r, int s, int len) {
    double* pc = double_malloc(len);
    double* apc = double_malloc(len);
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
    float* pc = float_malloc(len);
    float* apc = float_malloc(len);
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
    double* bp = double_malloc((len-1));
    double* tr = double_malloc((len-1));
    double* avgbp = double_malloc(len);
    double* avgtr = double_malloc(len);
    double* uo = double_malloc(len);

    _double_pairwise_min_unaligned(low+1, close, len-1, bp);
    _double_pairwise_max_unaligned(high+1, close, len-1, tr);
    _double_sub_arr(tr, bp, len - 1, tr);
    _double_sub_arr_unaligned(close+1, bp, len-1, bp);

    _double_running_sum(bp, len-1, s, avgbp);
    _double_running_sum(tr, len-1, s, avgtr);
    _double_div_arr_unaligned(avgbp, avgtr, len-1, uo+1);
    _double_mul(uo+1, len-1, ws, uo+1);

    _double_running_sum(bp, len-1, m, avgbp);
    _double_running_sum(tr, len-1, m, avgtr);
    _double_div_arr(avgbp, avgtr, len, avgbp);
    _double_mul(avgbp, len, wm, avgbp);
    _double_add_arr_unaligned(avgbp, uo+1, len-1, uo+1);

    _double_running_sum(bp, len-1, l, avgbp);
    _double_running_sum(tr, len-1, l, avgtr);
    _double_div_arr(avgbp, avgtr, len, avgbp);
    _double_mul(avgbp, len, wl, avgbp);
    _double_add_arr_unaligned(avgbp, uo+1, len-1, uo+1);

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
    float* bp = float_malloc((len-1));
    float* tr = float_malloc((len-1));
    float* avgbp = float_malloc(len);
    float* avgtr = float_malloc(len);
    float* uo = float_malloc(len);

    _float_pairwise_min_unaligned(low+1, close, len-1, bp);
    _float_pairwise_max_unaligned(high+1, close, len-1, tr);
    _float_sub_arr(tr, bp, len - 1, tr);
    _float_sub_arr_unaligned(close+1, bp, len-1, bp);

    _float_running_sum(bp, len-1, s, avgbp);
    _float_running_sum(tr, len-1, s, avgtr);
    _float_div_arr_unaligned(avgbp, avgtr, len-1, uo+1);
    _float_mul(uo+1, len-1, ws, uo+1);

    _float_running_sum(bp, len-1, m, avgbp);
    _float_running_sum(tr, len-1, m, avgtr);
    _float_div_arr(avgbp, avgtr, len, avgbp);
    _float_mul(avgbp, len, wm, avgbp);
    _float_add_arr_unaligned(avgbp, uo+1, len-1, uo+1);

    _float_running_sum(bp, len-1, l, avgbp);
    _float_running_sum(tr, len-1, l, avgtr);
    _float_div_arr(avgbp, avgtr, len, avgbp);
    _float_mul(avgbp, len, wl, avgbp);
    _float_add_arr_unaligned(avgbp, uo+1, len-1, uo+1);

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
    double* running_min = double_malloc(len);
    double* running_max = double_malloc(len);

    _double_running_max_unaligned(high, len, n, running_max);
    _double_running_min_unaligned(low, len, n, running_min);

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
    float* running_min = float_malloc(len);
    float* running_max = float_malloc(len);

    _float_running_max_unaligned(high, len, n, running_max);
    _float_running_min_unaligned(low, len, n, running_min);

    _float_sub_arr(running_max, running_min, len,
                    running_min);
    _float_sub_arr(running_max, close,
                    len, running_max);
    _float_div_arr(running_max, running_min, len, running_min);
    _float_mul(running_min, len, -100.f, running_min);

    free(running_max);

    return running_min;
}