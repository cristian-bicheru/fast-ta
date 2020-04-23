#pragma once

double* _ADI_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len);
float* _ADI_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len);

double* _CMF_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len, int n);
float* _CMF_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len, int n);

struct double_array_pair _EMV_DOUBLE(const double* high, const double* low,
                                     const double* volume, int len, int n);
struct float_array_pair _EMV_FLOAT(const float* high, const float* low,
                                   const float* volume, int len, int n);