#pragma once

/**
 * Computes Relative Strength Indicator On Data
 * @param close     Close Time Series
 * @param out       The array to output the Rsi Time Series into. Can be NULL
 *                  if you want this function to allocate the memory itself.
 * @param close_len Length of Close Time Series
 * @param _n        n-Value
 * @param prelim    The amount of values to compute before the first value in
 *                  close. WARNING: This will go before the close pointer, so
 *                  if there is no data there it will cause a SEFAULT. This
 *                  should only be done when there are enough values in close
 *                  to accomodate the prelim count.
 * @return          RSI Indicator Time Series
 */
double* _RSI_DOUBLE(const double* close, double* out, int close_len,
                    int window_size, int prelim);
float* _RSI_FLOAT(const float* close, float* out, int close_len,
                    int window_size, int prelim);

/**
 * Computes Awesome Oscillator Indicator On Data
 * @param high  High Time Series
 * @param low   Low Time Series     Note: high and low must have same lengths,
 *                                        this should be checked for before calling.
 * @param n1    High Time Series SMA Window Length
 * @param n2    Low Time Series SMA Window Length
 * @param len   Time Series Length
 * @return      AO Indicator Time Series
 */
double* _AO_DOUBLE(double* high, double* low, int n1, int n2, int len);
float* _AO_FLOAT(float * high, float * low, int n1, int n2, int len);

/**
 * Computes KAMA Indicator On Data
 * @param close     Close Time Series
 * @param n1        Length
 * @param n2        Fast Alpha
 * @param n3        Slow Alpha
 * @param len       Length of Close Time Series
 * @return          KAMA Indicator Time Series
 */
double* _KAMA_DOUBLE(double* close, int n1, int n2, int n3, int len);
float* _KAMA_FLOAT(float* close, int n1, int n2, int n3, int len);

/**
 * Compute ROC Indicator On Data
 * @param close     Close Time Series
 * @param n         Period
 * @param len       Close Time Series Length
 * @return          ROC Indicator Time Series
 */
double* _ROC_DOUBLE(double* close, int n, int len);
float* _ROC_FLOAT(float* close, int n, int len);

struct double_array_pair
_STOCHASTIC_OSCILLATOR_DOUBLE(double* high, double* low, double* close, int n,
                              int d, int len);
struct float_array_pair
_STOCHASTIC_OSCILLATOR_FLOAT(float* high, float* low, float* close, int n,
                             int d, int len);
