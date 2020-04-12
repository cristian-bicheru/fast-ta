#pragma once

/**
 * Computes Relative Strength Indicator On Data
 * @param close     Close Time Series
 * @param close_len Length of Close Time Series
 * @param _n        n-Value
 * @return          RSI Indicator Time Series
 */
double* _RSI_DOUBLE(const double* close, double* out, int close_len,
                    int window_size);
float* _RSI_FLOAT(const float* close, float* out, int close_len,
                    int window_size);

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
