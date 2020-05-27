#pragma once
#include <cmath>
#include "generic_simd.h"

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

const char file_format[25] = "%lf,%lf,%lf,%lf,%lf,%lf,";
const char file_format_fours[17] = "%lf,%lf,%lf,%lf,";
const int data_len = 253;

double get_abs_max(const double* arr, int len) {
    double m = fabs(arr[0]);
    for (int i = 1; i < len; i++) {
        m = max(m, fabs(arr[i]));
    }
    return m;
}

void load_data_sixes(const char* path, double* out) {
    FILE* f;
    char buf[1000];
    int i = 0;

    f = fopen(path, "r");
    if (f == nullptr) {
        std::cerr << "Unable to open path " << path << std::endl;
        exit(1);
    }

    while (fgets(buf, sizeof(buf), f)) {
        sscanf(buf, file_format, &out[i], &out[i+1], &out[i+2], &out[i+3], &out[i+4], &out[i+5]);
        i += 6;
    }

    fclose(f);
}

void load_data_fours(const char* path, double* out) {
    FILE* f;
    char buf[1000];
    int i = 0;

    f = fopen(path, "r");
    if (f == nullptr) {
        std::cerr << "Unable to open path " << path;
        exit(1);
    }

    while (fgets(buf, sizeof(buf), f)) {
        sscanf(buf, file_format_fours, &out[i], &out[i+1], &out[i+2], &out[i+3]);
        i += 4;
    }

    fclose(f);
}

// this is the largest the floating point error can be.
// there are deltas seen between float calculations and double calculations.
// this is that max delta.
// computed as precision * length of dataset, this ensures that even if floating
// point errors add up, it should remain bounded by the max_fp_error
double get_max_fp_error(const double* x, int len) {
    return get_abs_max(x, len)*len*FLT_EPSILON;
}

double get_max_dp_error(const double* x, int len) {
    return get_abs_max(x, len)*len*DBL_EPSILON;
}

alignas(sizeof(double)*DOUBLE_VEC_SIZE) double SAMPLE_OPEN_DOUBLE[data_len] = {0};
alignas(sizeof(float)*FLOAT_VEC_SIZE) float SAMPLE_OPEN_FLOAT[data_len] = {0};

alignas(sizeof(double)*DOUBLE_VEC_SIZE) double SAMPLE_CLOSE_DOUBLE[data_len] = {0};
alignas(sizeof(float)*FLOAT_VEC_SIZE) float SAMPLE_CLOSE_FLOAT[data_len] = {0};

alignas(sizeof(double)*DOUBLE_VEC_SIZE) double SAMPLE_HIGH_DOUBLE[data_len] = {0};
alignas(sizeof(float)*FLOAT_VEC_SIZE) float SAMPLE_HIGH_FLOAT[data_len] = {0};

alignas(sizeof(double)*DOUBLE_VEC_SIZE) double SAMPLE_LOW_DOUBLE[data_len] = {0};
alignas(sizeof(float)*FLOAT_VEC_SIZE) float SAMPLE_LOW_FLOAT[data_len] = {0};

alignas(sizeof(double)*DOUBLE_VEC_SIZE) double SAMPLE_VOLUME_DOUBLE[data_len] = {0};
alignas(sizeof(float)*FLOAT_VEC_SIZE) float SAMPLE_VOLUME_FLOAT[data_len] = {0};

void populate_float_arrays() {
    // init all the float arrays from the double arrays
    load_data_sixes("fast_ta/src/test_data/sample_close_data.txt",
                    SAMPLE_CLOSE_DOUBLE);
    load_data_sixes("fast_ta/src/test_data/sample_high_data.txt",
                    SAMPLE_HIGH_DOUBLE);
    load_data_sixes("fast_ta/src/test_data/sample_low_data.txt",
                    SAMPLE_LOW_DOUBLE);
    load_data_sixes("fast_ta/src/test_data/sample_open_data.txt",
                    SAMPLE_OPEN_DOUBLE);
    load_data_sixes("fast_ta/src/test_data/sample_volume_data.txt",
                    SAMPLE_VOLUME_DOUBLE);
    for (int i=0; i<data_len; i++) {
        SAMPLE_OPEN_FLOAT[i] = (float)SAMPLE_OPEN_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        SAMPLE_CLOSE_FLOAT[i] = (float)SAMPLE_CLOSE_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        SAMPLE_HIGH_FLOAT[i] = (float)SAMPLE_HIGH_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        SAMPLE_LOW_FLOAT[i] = (float)SAMPLE_LOW_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        SAMPLE_VOLUME_FLOAT[i] = (float)SAMPLE_VOLUME_DOUBLE[i];
    }
}