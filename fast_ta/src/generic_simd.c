#include "generic_simd.h"

inline __attribute__((always_inline)) int double_get_next_index(int len, int start) {
    return len - (len-start) % DOUBLE_VEC_SIZE;
}

inline __attribute__((always_inline)) int float_get_next_index(int len, int start) {
    return len - (len-start) % FLOAT_VEC_SIZE;
}

double* double_malloc(int len) {
    return aligned_alloc(sizeof(double)*DOUBLE_VEC_SIZE, len* sizeof(double));
}

float* float_malloc(int len) {
    return aligned_alloc(sizeof(float)*FLOAT_VEC_SIZE, len* sizeof(float));
}

bool check_double_align(const double* arr) {
    return ((uint64_t) arr)%(sizeof(double)*DOUBLE_VEC_SIZE) == 0 ? true : false;
}

bool check_float_align(const float* arr) {
    return ((uint64_t) arr)%(sizeof(float)*FLOAT_VEC_SIZE) == 0 ? true : false;
}

STATIC_ASSERT(sizeof(double) * DOUBLE_VEC_SIZE == sizeof(float) * FLOAT_VEC_SIZE, INCONSISTENT_VECTOR_WIDTHS);
const uint64_t ALGN_MOD = (sizeof(double) * DOUBLE_VEC_SIZE)-1;

int64_t double_next_aligned_pointer(const double* addr) {
    return DOUBLE_VEC_SIZE - (((uint64_t) addr & ALGN_MOD) / sizeof(double));
}

int64_t float_next_aligned_pointer(const float* addr) {
    return FLOAT_VEC_SIZE - (((uint64_t) addr & ALGN_MOD) / sizeof(float));
}

const float fltmax[8] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
const float nfltmax[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
const double dblmax[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
const double ndblmax[4] = {-DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX};