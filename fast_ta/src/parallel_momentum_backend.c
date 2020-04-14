/**
 * This file uses the functions defined in momentum_backend.c and calls them
 * in parallel.
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#include "error_methods.h"
#include "momentum_backend_args.h"
#include "momentum_backend.h"

/*
 * Only meant as a private helper. Not defined in header.
 */
void* _rsi_double_helper(void* _args) {
    struct RsiDoubleArgs* args = (struct RsiDoubleArgs*)_args;
    _RSI_DOUBLE(args->close, args->out, args->close_len, args->window_size,
                args->prelim);

    pthread_exit(NULL);
}

double* _PARALLEL_RSI_DOUBLE(const double* close, int close_len,
                             int window_size, int thread_count) {
    double* rsi = malloc(close_len * sizeof(double));
    // zero the dynamic memory so it is obvious when values are not filled in
    for (int i=0; i<close_len; i++) {
        rsi[i] = 0;
    }

    pthread_t threads[thread_count];
    struct RsiDoubleArgs args[thread_count];

    // the +1 here is to accomodate for uneven division of the Close Time Series.
    int chunk_size = close_len/thread_count + 1;

    // the rest of the workers are normal
    for (int i=0; i<thread_count; i++) {
        int offset = i * chunk_size;

        args[i].close = close + offset;
        args[i].out = rsi + offset;
        args[i].window_size = window_size;
        // TODO: This could throw an error if the window_size pushes the pointer
        // behind the array. Also, there is an exception for the first chunk
        // as there is no data to look at before it.
        args[i].prelim = i == 0 ? 0 : window_size;
        if (offset + chunk_size + 1 > close_len) {
            // if we are at the end of the array, just compute to the end of
            // the array and don't go out of bounds.
            args[i].close_len = close_len - offset;
        } else {
            // the +1 here is to compute an overlapping value with the next
            // block since each block's first value is messed up.
            args[i].close_len = chunk_size + 1;
        }

        int error = pthread_create(&threads[i], NULL, _rsi_double_helper,
                                   (void *) &args[i]);
        if (error != 0) {
            raise_error("Error creating pthread.");
            return NULL;
        }
    }

    for (int i=0; i<thread_count; i++) {
        int error = pthread_join(threads[i], NULL);
        if (error != 0) {
            raise_error("Error joining pthread.");
            return NULL;
        }
    }

    printf("done\n");

    return rsi;
}

/*
 * Only meant as a private helper. Not defined in header.
 */
void* _rsi_float_helper(void* _args) {
    struct RsiFloatArgs* args = (struct RsiFloatArgs*)_args;
    _RSI_FLOAT(args->close, args->out, args->close_len, args->window_size,
                args->prelim);

    pthread_exit(NULL);
}

float* _PARALLEL_RSI_FLOAT(const float* close, int close_len,
                             int window_size, int thread_count) {
    float* rsi = malloc(close_len * sizeof(float));
    // zero the dynamic memory so it is obvious when values are not filled in
    for (int i=0; i<close_len; i++) {
        rsi[i] = 0;
    }

    pthread_t threads[thread_count];
    struct RsiFloatArgs args[thread_count];

    // the +1 here is to accomodate for uneven division of the Close Time Series.
    int chunk_size = close_len/thread_count + 1;

    // the rest of the workers are normal
    for (int i=0; i<thread_count; i++) {
        int offset = i * chunk_size;

        args[i].close = close + offset;
        args[i].out = rsi + offset;
        args[i].window_size = window_size;
        // TODO: This could throw an error if the window_size pushes the pointer
        // behind the array. Also, there is an exception for the first chunk
        // as there is no data to look at before it.
        args[i].prelim = i == 0 ? 0 : window_size;
        if (offset + chunk_size + 1 > close_len) {
            // if we are at the end of the array, just compute to the end of
            // the array and don't go out of bounds.
            args[i].close_len = close_len - offset;
        } else {
            // the +1 here is to compute an overlapping value with the next
            // block since each block's first value is messed up.
            args[i].close_len = chunk_size + 1;
        }

        int error = pthread_create(&threads[i], NULL, _rsi_float_helper,
                                   (void *) &args[i]);
        if (error != 0) {
            raise_error("Error creating pthread.");
            return NULL;
        }
    }

    for (int i=0; i<thread_count; i++) {
        int error = pthread_join(threads[i], NULL);
        if (error != 0) {
            raise_error("Error joining pthread.");
            return NULL;
        }
    }

    printf("done\n");

    return rsi;
}
