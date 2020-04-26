#pragma once

double* _ADI_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len);
float* _ADI_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len);

double* _CMF_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len, int n);
float* _CMF_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len, int n);

double** _EMV_DOUBLE(const double* high, const double* low,
                                     const double* volume, int len, int n);
float** _EMV_FLOAT(const float* high, const float* low,
                                   const float* volume, int len, int n);

double* _FI_DOUBLE(const double* close, const double* volume, int len, int n);
float* _FI_FLOAT(const float* close, const float* volume, int len, int n);

double* _MFI_DOUBLE(const double* high, const double* low, const double* close,
                    const double* volume, int len, int n);
float* _MFI_FLOAT(const float* high, const float* low, const float* close,
                  const float* volume, int len, int n);

double* _NVI_DOUBLE(const double* close, const double* volume, int len);
float* _NVI_FLOAT(const float* close, const float* volume, int len);

double* _OBV_DOUBLE(const double* close, const double* volume, int len);
float* _OBV_FLOAT(const float* close, const float* volume, int len);

double* _VPT_DOUBLE(const double* close, const double* volume, int len);
float* _VPT_FLOAT(const float* close, const float* volume, int len);

double* _VWAP_DOUBLE(const double* high, const double* low, const double* close,
                     const double* volume, int len, int n);
float* _VWAP_FLOAT(const float* high, const float* low, const float* close,
                   const float* volume, int len, int n);