#include "generic_simd.h"

inline __attribute__((always_inline)) int double_get_next_index(int len, int start) {
    return len - (len-start) % DOUBLE_VEC_SIZE;
}

inline __attribute__((always_inline)) int float_get_next_index(int len, int start) {
    return len - (len-start) % FLOAT_VEC_SIZE;
}

const float fltmax[8] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
const float nfltmax[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
const double dblmax[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
const double ndblmax[4] = {-DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX};