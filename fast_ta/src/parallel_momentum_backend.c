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
void _rsi_double_helper(void* _args) {
    struct RsiDoubleArgs* args = (struct RsiDoubleArgs*)_args;
    _RSI_DOUBLE(args->close, args->out, args->close_len, args->window_size,
                args->prelim);
}

double* _PARALLEL_RSI_DOUBLE(const double* close, int close_len,
                             int window_size, int thread_count) {
    double* rsi = malloc(close_len * sizeof(double));

    pthread_t threads[thread_count];
    struct RsiDoubleArgs args[thread_count];

    int chunk_size = close_len/thread_count+1;
    // 1 is a special case since we _don't_ want to skip_perlim, since there
    // is no preliminary data to be calculated.
    args[0].close = close;
    args[0].out = rsi;
    args[0].close_len = chunk_size;
    args[0].window_size = window_size;
    args[0].prelim = 0;
    int error = pthread_create(&threads[0], NULL, _rsi_double_helper,
                               (void *) &args[0]);
    if (error != 0) {
        raise_error("Error creating pthread.");
        return NULL;
    }

    // the rest of the workers are normal
    for (int i=1; i<thread_count; i++) {
        int offset = i * chunk_size;
	if (offset+chunk_size>close_len) {
            offset = close_len-chunk_size;
        }
        args[i].close = close + offset;
        args[i].out = rsi + offset;
        args[i].close_len = chunk_size;
        args[i].window_size = window_size;

        // TODO: This could throw an error if the window_size pushes the pointer
        // behind the array
        args[i].prelim = window_size;

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
