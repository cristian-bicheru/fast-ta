#include <stdlib.h>
#include "funcs.c"

/**
 * Computes Relative Strength Indicator On Data
 * @param close     Close Time Series
 * @param close_len Length of Close Time Series
 * @param _n        n-Value
 * @return          RSI Indicator Time Series
 */
double* _RSI_DOUBLE(const double* close, int close_len, int _n) {
    double last_gain, last_loss, cd, cgain, closs;
    double* rsi = malloc(close_len * sizeof(double));

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

float* _RSI_FLOAT(const float* close, int close_len, int _n) {
    float last_gain, last_loss, cd, cgain, closs;
    float* rsi = malloc(close_len * sizeof(float));

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
 * Computes Awesome Oscillator Indicator On Data
 * @param high  High Time Series
 * @param low   Low Time Series  Note: high and low must have same lengths,
 *                                     his should be checked for before calling.
 * @param n1    High Time Series SMA Window Length
 * @param n2    Low Time Series SMA Window Length
 * @param len   Time Series Length
 * @return      AO Indicator Time Series
 */
double* _AO_DOUBLE(double*  high, double*  low, int n1, int n2, int len) {
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
