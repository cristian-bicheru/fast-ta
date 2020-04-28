#pragma once

double* _ATR_DOUBLE(const double* high, const double* low, const double* close,
                    int len, int n);
float* _ATR_FLOAT(const float* high, const float* low, const float* close,
                  int len, int n);

double** _BOL_DOUBLE(const double* close, int len, int n, float ndev);
float** _BOL_FLOAT(const float* close, int len, int n, float ndev);

double** _DC_DOUBLE(const double* high, const double* low, int len, int n);
float** _DC_FLOAT(const float* high, const float* low, int len, int n);

double** _KC_DOUBLE(const double* high, const double* low, const double* close,
                    int len, int n1, int n2, int num_channels);
float** _KC_FLOAT(const float* high, const float* low, const float* close,
                  int len, int n1, int n2, int num_channels);