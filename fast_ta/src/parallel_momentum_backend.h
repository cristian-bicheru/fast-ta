/**
 * This file uses the functions defined in momentum_backend.c and calls them
 * in parallel.
 */
#pragma once

#include "momentum_backend_args.h"
#include "momentum_backend.h"

/**
 * Spawns worker threads to run _RSI_DOUBLE in parallel.
 * @param close        Array of closes over time.
 * @param close_len    The length of the closes over time.
 * @param window_size  The size of the moving average window to use.
 * @param thread_count The number of threads to use.
 * @return             Returns a pointer to the dynamically allocated RSI Time
 *                     Series
 */
double* _PARALLEL_RSI_DOUBLE(const double* close, int close_len,
                             int window_size, int thread_count);
float* _PARALLEL_RSI_FLOAT(const float* close, int close_len,
                             int window_size, int thread_count);
