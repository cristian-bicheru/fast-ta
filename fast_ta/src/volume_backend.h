#pragma once

double* _ADI_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len);
float* _ADI_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len);