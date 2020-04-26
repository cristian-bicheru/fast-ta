#pragma once

enum stoch_mode {Normal, Williams};

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
double* _AO_DOUBLE(const double* high, const double* low, int n1, int n2, int len);
float* _AO_FLOAT(const float * high, const float * low, int n1, int n2, int len);

/**
 * Computes KAMA Indicator On Data
 * @param close     Close Time Series
 * @param n1        Length
 * @param n2        Fast Alpha
 * @param n3        Slow Alpha
 * @param len       Length of Close Time Series
 * @return          KAMA Indicator Time Series
 */
double* _KAMA_DOUBLE(const double* close, int n1, int n2, int n3, int len);
float* _KAMA_FLOAT(const float* close, int n1, int n2, int n3, int len);

/**
 * Compute ROC Indicator On Data
 * @param close     Close Time Series
 * @param n         Period
 * @param len       Close Time Series Length
 * @return          ROC Indicator Time Series
 */
double* _ROC_DOUBLE(const double* close, int n, int len);
float* _ROC_FLOAT(const float* close, int n, int len);

/**
 * Compute Stochastic Oscillator On Data
 * @param high      High Time Series
 * @param low       Low Time Series
 * @param close     Close Time Series
 * @param n         n Value
 * @param d         d Value
 * @param len       Time Series Length
 * @param mode      Compute SMA Flag
 * @return          Stochastic Oscillator Indicator Time Series And Signal If
 *                  Specified
 */
double**
_STOCHASTIC_OSCILLATOR_DOUBLE(const double* high, const double* low, double* close, int n,
                              int d, int len, enum stoch_mode mode);
float**
_STOCHASTIC_OSCILLATOR_FLOAT(const float* high, const float* low, float* close, int n,
                             int d, int len, enum stoch_mode mode);

/**
 * Compute True Strength Index On Data
 * @param close     Close Time Series
 * @param r         R Value
 * @param s         S Value
 * @param len       Close Time Series Length
 * @return          TSI Indicator Time Series
 */
double* _TSI_DOUBLE(const double* close, int r, int s, int len);
float* _TSI_FLOAT(const float* close, int r, int s, int len);

/**
 * Compute Ultimate Oscillator On Data
 * @param high      High Time Series
 * @param low       Low Time Series
 * @param close     Close Time Series
 * @param s         Short Period
 * @param m         Medium Period
 * @param l         Long Period
 * @param ws        Short Period Weight
 * @param wm        Medium Period Weight
 * @param wl        Long Period Weight
 * @param len       Time Series Length
 * @return          Ultimate Oscillator Time Series
 */
double* _ULTIMATE_OSCILLATOR_DOUBLE(const double* high, const double* low, const double* close, int s, int m, int l, double ws, double wm, double wl, int len);
float* _ULTIMATE_OSCILLATOR_FLOAT(const float* high, const float* low, const float* close, int s, int m, int l, double ws, double wm, double wl, int len);

/**
 * Compute Williams %R
 * @param high      High Time Series
 * @param low       Low Time Series
 * @param close     Close Time Series
 * @param n         n Value
 * @param d         d Value
 * @param len       Time Series Length
 * @return          Williams %R Time Series
 */
double* _WILLIAMS_R_DOUBLE(const double* high, const double* low, double* close, int n,
                           int len);
float* _WILLIAMS_R_FLOAT(const float* high, const float* low, float* close, int n,
                           int len);