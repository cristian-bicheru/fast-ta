#include <stdlib.h>
#include <stdio.h>

#include "funcs.h"
#include "momentum_backend.h"

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
    } else {
        // some funky pointer math to move the close back the specified amount
        // so we can get the moving averages to where they should be.
        // NOTE: This is the dangerous part of the code.
        double* reduced_close = close - prelim;
        for (int i = 0; i<prelim; i++) {
            double cd = reduced_close[i+1]-reduced_close[i];

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
        }
    }

    int start = prelim == 0 ? _n+1 : 0;
    for (int i = start; i < close_len; i++) {
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

float* _RSI_FLOAT(const float* close, float* out, int close_len, int _n,
                  int prelim) {
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
