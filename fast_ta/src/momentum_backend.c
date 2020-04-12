#include <stdlib.h>

#include "funcs.c"
#include "momentum_backend.h"

double* _RSI_DOUBLE(const double* close, double* out, int close_len, int _n) {
    // support the user mallocing themselves OR this function allocating the memory
    double* rsi;
    if (out == NULL) {
        rsi = malloc(close_len * sizeof(double));
    } else {
        rsi = out;
    }

    double last_gain = 0;
    double last_loss = 0;
    for (int i = 0; i<_n; i++) {
        double cd = close[i+1]-close[i];

        double cgain;
        double closs;
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

        last_gain = (last_gain*i+cgain)/(i+1);
        last_loss = (last_loss*i+closs)/(i+1);

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
        double cd = close[i]-close[i-1];

        double cgain;
        double closs;
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

        last_gain = (last_gain*13+cgain)/14;
        last_loss = (last_loss*13+closs)/14;

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

/**
 * Computes Relative Strength Indicator On Data
 * @param close     Close Time Series
 * @param close_len Length of Close Time Series
 * @param _n        n-Value
 * @return          RSI Indicator Time Series
 */
float* _RSI_FLOAT(const float* close, float* out, int close_len, int _n) {
    // support the user mallocing themselves OR this function allocating the memory
    float* rsi;
    if (out == NULL) {
        rsi = malloc(close_len * sizeof(float));
    } else {
        rsi = out;
    }

    float last_gain = 0;
    float last_loss = 0;
    for (int i = 0; i<_n; i++) {
        float cd = close[i+1]-close[i];

        float cgain;
        float closs;
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

        last_gain = (last_gain*i+cgain)/(i+1);
        last_loss = (last_loss*i+closs)/(i+1);

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
        float cd = close[i]-close[i-1];

        float cgain;
        float closs;
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

        last_gain = (last_gain*13+cgain)/14;
        last_loss = (last_loss*13+closs)/14;

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

double* _AO_DOUBLE(double* high, double* low, int n1, int n2, int len) {
    double* median = _double_pairwise_mean(high, low, len);
    double* sma1 = _double_sma(median, len, n1);
    double* sma2 = _double_sma(median, len, n2);
    _double_sub(sma1, sma2, median, len);
    return median;
}

float* _AO_FLOAT(float* high, float* low, int n1, int n2, int len) {
    float* median = _float_pairwise_mean(high, low, len);
    float* sma1 = _float_sma(median, len, n1);
    float* sma2 = _float_sma(median, len, n2);
    _float_sub(sma1, sma2, median, len);
    return median;
}
