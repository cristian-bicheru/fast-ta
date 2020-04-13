#include <stdlib.h>
#include "funcs.c"
#include "debug_tools.c"

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

/**
 * Computes KAMA Indicator On Data
 * @param close     Close Time Series
 * @param n1        Length
 * @param n2        Fast Alpha
 * @param n3        Slow Alpha
 * @param len       Length of Close Time Series
 * @return
 */
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