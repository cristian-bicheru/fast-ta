#include "core_backend.h"
#include "../generic_simd/generic_simd.h"

double* _ALIGN_DOUBLE(const double* data, int len) {
    double* ret = double_malloc(len);

    for (int i = 0; i <= len - DOUBLE_VEC_SIZE; i += DOUBLE_VEC_SIZE) {
        _double_store(&ret[i], _double_loadu(&data[i]));
    }

    for (int i = double_get_next_index(len, 0); i < len; i++) {
        ret[i] = data[i];
    }

    return ret;
}

float* _ALIGN_FLOAT(const float* data, int len) {
    float* ret = float_malloc(len);

    for (int i = 0; i <= len - FLOAT_VEC_SIZE; i += FLOAT_VEC_SIZE) {
        _float_store(&ret[i], _float_loadu(&data[i]));
    }

    for (int i = float_get_next_index(len, 0); i < len; i++) {
        ret[i] = data[i];
    }

    return ret;
}
