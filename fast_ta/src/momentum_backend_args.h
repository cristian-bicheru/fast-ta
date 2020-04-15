#pragma once

/**
 * We must define structs to contain the args to the backend functions that 
 * will be parallelized. This way the arguments can be placed in an array.
 */

/**
 * Contains the parameters to the _RSI_DOUBLE backend functions.
 */
struct RSIDoubleArgs {
    const double* close;
    double* out;
    int close_len;
    int window_size;
    int prelim;
};

/**
 * Contains the parameters to the _RSI_FLOAT backend functions.
 */
struct RSIFloatArgs {
    const float* close;
    float* out;
    int close_len;
    int window_size;
    int prelim;
};
