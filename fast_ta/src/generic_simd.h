#pragma once
#include <math.h>
#include <float.h>

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

/**
 * Find The First Index Not Covered By Vector
 * @param len
 * @param start
 * @return
 */
int float_get_next_index(int len, int start);
int double_get_next_index(int len, int start);

/** Generic SIMD Support **/

#ifdef AVX
/** AVX Support **/
    #include <immintrin.h>

    #define __float_vector __m256
    #define __double_vector __m256d
    #define __int_vector __m256i
    #define FLOAT_VEC_SIZE 8
    #define DOUBLE_VEC_SIZE 4
    #define simd_malloc(size) (alligned_malloc(256, size))

    extern const float fltmax[8];
    extern const float nfltmax[8];
    extern const double dblmax[4];
    extern const double ndblmax[4];

    inline __attribute__((always_inline)) void _float_storeu(float* addr, const __float_vector A) {
        _mm256_storeu_ps(addr, A);
    }

    inline __attribute__((always_inline)) void _double_storeu(double* addr, const __double_vector A) {
        _mm256_storeu_pd(addr, A);
    }

    inline __attribute__((always_inline)) __float_vector _float_loadu(const float* addr) {
        return _mm256_loadu_ps(addr);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu(const double* addr) {
        return _mm256_loadu_pd(addr);
    }

    inline __attribute__((always_inline)) __int_vector _int_loadu(const void* addr) {
        return _mm256_loadu_si256((__int_vector*) addr);
    }

    inline __attribute__((always_inline)) __float_vector _float_maskload(const float* addr, const __int_vector mask) {
        return _mm256_maskload_ps(addr, mask);
    }

    inline __attribute__((always_inline)) __double_vector _double_maskload(const double* addr, const __int_vector mask) {
        return _mm256_maskload_pd(addr, mask);
    }

    inline __attribute__((always_inline)) void _float_maskstore(float* addr, const __int_vector mask, const __float_vector A) {
        _mm256_maskstore_ps(addr, mask, A);
    }

    inline __attribute__((always_inline)) void _double_maskstore(double* addr, const __int_vector mask, const __double_vector A) {
        _mm256_maskstore_pd(addr, mask, A);
    }

    inline __attribute__((always_inline)) __float_vector _float_add_vec(__float_vector A, __float_vector B) {
        return _mm256_add_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_add_vec(__double_vector A, __double_vector B) {
        return _mm256_add_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_sub_vec(__float_vector A, __float_vector B) {
        return _mm256_sub_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_sub_vec(__double_vector A, __double_vector B) {
        return _mm256_sub_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mul_vec(__float_vector A, __float_vector B) {
        return _mm256_mul_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_mul_vec(__double_vector A, __double_vector B) {
        return _mm256_mul_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_div_vec(__float_vector A, __float_vector B) {
        return _mm256_mul_ps(A, _mm256_rcp_ps(B));
    }

    inline __attribute__((always_inline)) __double_vector _double_div_vec(__double_vector A, __double_vector B) {
        return _mm256_div_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_set1_vec(float a) {
        return _mm256_set1_ps(a);
    }

    inline __attribute__((always_inline)) __double_vector _double_set1_vec(double a) {
        return _mm256_set1_pd(a);
    }

    inline __attribute__((always_inline)) __float_vector _float_set_vec(float a, float b, float c, float d, float e, float f, float g, float h) {
        return _mm256_set_ps(a, b, c, d, e, f, g, h);
    }

    inline __attribute__((always_inline)) __double_vector _double_set_vec(double a, double b, double c, double d) {
        return _mm256_set_pd(a, b, c, d);
    }

    inline __attribute__((always_inline)) __float_vector _float_abs_vec(__float_vector x, __float_vector sign_mask) {
        return _mm256_andnot_ps(sign_mask, x);
    }

    inline __attribute__((always_inline)) __double_vector _double_abs_vec(__double_vector x, __double_vector sign_mask) {
        return _mm256_andnot_pd(sign_mask, x);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu2(const double* A, const double* B) {
        return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(A)), _mm_loadu_pd(B), 1);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu2_from_float(const float* A, const float* B) {
        return _double_loadu((double[4]) {A[0], A[1], B[0], B[1]});
    }

    inline __attribute__((always_inline)) __float_vector _float_max_vec(__float_vector A, __float_vector B) {
        return _mm256_max_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_max_vec(__double_vector A, __double_vector B) {
        return _mm256_max_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_max_vec(__float_vector A, __float_vector B, __int_vector mask) {
        return _mm256_blendv_ps(_mm256_loadu_ps(nfltmax), _mm256_max_ps(A, B), (__m256) mask);
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_max_vec(__double_vector A, __double_vector B, __int_vector mask) {
        return _mm256_blendv_pd(_mm256_loadu_pd(ndblmax), _mm256_max_pd(A, B), (__m256d) mask);
    }

    inline __attribute__((always_inline)) __float_vector _float_min_vec(__float_vector A, __float_vector B) {
        return _mm256_min_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(__double_vector A, __double_vector B) {
        return _mm256_min_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_min_vec(__float_vector A, __float_vector B, __int_vector mask) {
        return _mm256_blendv_ps(_mm256_loadu_ps(fltmax), _mm256_min_ps(A, B), (__m256) mask);
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_min_vec(__double_vector A, __double_vector B, __int_vector mask) {
        return _mm256_blendv_pd(_mm256_loadu_pd(dblmax), _mm256_min_pd(A, B), (__m256d) mask);
    }

    inline __attribute__((always_inline)) __float_vector _float_setzero_vec() {
        return _mm256_setzero_ps();
    }

    inline __attribute__((always_inline)) __double_vector _double_setzero_vec() {
        return _mm256_setzero_pd();
    }

    inline __attribute__((always_inline)) float _float_index_vec(const __float_vector A, const int i) {
        return A[i];
    }

    inline __attribute__((always_inline)) double _double_index_vec(const __double_vector A, const int i) {
        return A[i];
    }

    inline __attribute__((always_inline)) __float_vector _float_recp_vec(const __float_vector A) {
        return _mm256_rcp_ps(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_recp_vec(const __double_vector A) {
        return _mm256_div_pd(_mm256_set1_pd(1.), A);
    }

    inline __attribute__((always_inline)) __float_vector _float_rsqrt_vec(const __float_vector A) {
        return _mm256_rsqrt_ps(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_rsqrt_vec(const __double_vector A) {
        return _mm256_div_pd(_mm256_set1_pd(1.), _mm256_sqrt_pd(A));
    }

    inline __attribute__((always_inline)) __float_vector _float_sqrt_vec(const __float_vector A) {
        return _mm256_rcp_ps(_mm256_rsqrt_ps(A));
    }

    inline __attribute__((always_inline)) __double_vector _double_sqrt_vec(const __double_vector A) {
        return _mm256_sqrt_pd(A);
    }

#elif defined(SSE2)
/** SSE Support **/
    #include <immintrin.h>
    #define __float_vector __m128
    #define __double_vector __m128d
    #define __int_vector __m128i
    #define FLOAT_VEC_SIZE 4
    #define DOUBLE_VEC_SIZE 2
    #define simd_malloc(size) (alligned_malloc(128, size))

    inline __attribute__((always_inline)) void _float_storeu(float* addr, const __float_vector A) {
        _mm_storeu_ps(addr, A);
    }

    inline __attribute__((always_inline)) void _double_storeu(double* addr, const __double_vector A) {
        _mm_storeu_pd(addr, A);
    }

    inline __attribute__((always_inline)) __float_vector _float_loadu(const float* addr) {
        return _mm_loadu_ps(addr);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu(const double* addr) {
        return _mm_loadu_pd(addr);
    }

    inline __attribute__((always_inline)) __int_vector _int_loadu(const void* addr) {
        return _mm_loadu_si128((__int_vector*) addr);
    }

    inline __attribute__((always_inline)) __float_vector _float_maskload(const float* addr, const __int_vector mask) {
        return _mm_set_ps(mask[0] != 0 ? addr[0] : 0,
                          mask[32] != 0 ? addr[1] : 0,
                          mask[64] != 0 ? addr[2] : 0,
                          mask[96] != 0 ? addr[3] : 0);
    }

    inline __attribute__((always_inline)) __double_vector _double_maskload(const double* addr, const __int_vector mask) {
        return _mm_set_pd(mask[0] != 0 ? addr[0] : 0,
                          mask[32] != 0 ? addr[1] : 0);
    }

    inline __attribute__((always_inline)) void _float_maskstore(float* addr, const __int_vector mask, const __float_vector A) {
        for (int i = 0; i < FLOAT_VEC_SIZE; i++) {
            if (mask[i*32] != 0) {
                addr[i] = A[i];
            }
        }
    }

    inline __attribute__((always_inline)) void _double_maskstore(double* addr, const __int_vector mask, const __double_vector A) {
        for (int i = 0; i < DOUBLE_VEC_SIZE; i++) {
            if (mask[i*32] != 0) {
                addr[i] = A[i];
            }
        }
    }

    inline __attribute__((always_inline)) __float_vector _float_add_vec(__float_vector A, __float_vector B) {
        return _mm_add_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_add_vec(__double_vector A, __double_vector B) {
        return _mm_add_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_sub_vec(__float_vector A, __float_vector B) {
        return _mm_sub_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_sub_vec(__double_vector A, __double_vector B) {
        return _mm_sub_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mul_vec(__float_vector A, __float_vector B) {
        return _mm_mul_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_mul_vec(__double_vector A, __double_vector B) {
        return _mm_mul_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_div_vec(__float_vector A, __float_vector B) {
        return _mm_div_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_div_vec(__double_vector A, __double_vector B) {
        return _mm_div_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_set1_vec(float a) {
        return _mm_set1_ps(a);
    }

    inline __attribute__((always_inline)) __double_vector _double_set1_vec(double a) {
        return _mm_set1_pd(a);
    }

    inline __attribute__((always_inline)) __float_vector _float_set_vec(float a, float b, float c, float d) {
        return _mm_set_ps(a, b, c, d);
    }

    inline __attribute__((always_inline)) __double_vector _double_set_vec(double a, double b) {
        return _mm_set_pd(a, b);
    }

    inline __attribute__((always_inline)) __float_vector _float_abs_vec(__float_vector x, __float_vector sign_mask) {
        return _mm_andnot_ps(sign_mask, x);
    }

    inline __attribute__((always_inline)) __double_vector _double_abs_vec(__double_vector x, __double_vector sign_mask) {
        return _mm_andnot_pd(sign_mask, x);
    }

    inline __attribute__((always_inline)) __float_vector _float_max_vec(__float_vector A, __float_vector B) {
        return _mm_max_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_max_vec(__double_vector A, __double_vector B) {
        return _mm_max_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_max_vec(__float_vector A, __float_vector B, __int_vector mask) {
        __m128 v = _mm_max_ps(A, B);
        return _mm_set_ps(mask[0] != 0 ? v[0] : -FLT_MAX,
                          mask[32] != 0 ? v[1] : -FLT_MAX,
                          mask[64] != 0 ? v[2] : -FLT_MAX,
                          mask[96] != 0 ? v[3] : -FLT_MAX);
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_max_vec(__double_vector A, __double_vector B, __int_vector mask) {
        __m128d v = _mm_max_pd(A, B);
        return _mm_set_pd(mask[0] != 0 ? v[0] : -DBL_MAX,
                          mask[32] != 0 ? v[1] : -DBL_MAX);
    }

    inline __attribute__((always_inline)) __float_vector _float_min_vec(__float_vector A, __float_vector B) {
        return _mm_min_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(__double_vector A, __double_vector B) {
        return _mm_min_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_min_vec(__float_vector A, __float_vector B, __int_vector mask) {
        __m128 v = _mm_min_ps(A, B);
        return _mm_set_ps(mask[0] != 0 ? v[0] : FLT_MAX,
                          mask[32] != 0 ? v[1] : FLT_MAX,
                          mask[64] != 0 ? v[2] : FLT_MAX,
                          mask[96] != 0 ? v[3] : FLT_MAX);
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_min_vec(__double_vector A, __double_vector B, __int_vector mask) {
        __m128d v = _mm_min_pd(A, B);
        return _mm_set_pd(mask[0] != 0 ? v[0] : DBL_MAX,
                          mask[32] != 0 ? v[1] : DBL_MAX);
    }

    inline __attribute__((always_inline)) __float_vector _float_setzero_vec() {
        return _mm_setzero_ps();
    }

    inline __attribute__((always_inline)) __double_vector _double_setzero_vec() {
        return _mm_setzero_pd();
    }

    inline __attribute__((always_inline)) float _float_index_vec(const __float_vector A, const int i) {
        return A[i];
    }

    inline __attribute__((always_inline)) double _double_index_vec(const __double_vector A, const int i) {
        return A[i];
    }

    inline __attribute__((always_inline)) __float_vector _float_recp_vec(const __float_vector A) {
        return _mm_rcp_ps(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_recp_vec(const __double_vector A) {
        return _mm_div_pd(_mm_set1_pd(1.), A);
    }

    inline __attribute__((always_inline)) __float_vector _float_rsqrt_vec(const __float_vector A) {
        return _mm_rsqrt_ps(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_rsqrt_vec(const __double_vector A) {
        return _mm_div_pd(_mm_set1_pd(1.), _mm_sqrt_pd(A));
    }

    inline __attribute__((always_inline)) __float_vector _float_sqrt_vec(const __float_vector A) {
        return _mm_rcp_ps(_mm_rsqrt_ps(A));
    }

    inline __attribute__((always_inline)) __double_vector _double_sqrt_vec(const __double_vector A) {
        return _mm_sqrt_pd(A);
    }

#elif defined(AVX512)
/** AVX512 Support **/
    #include <immintrin.h>

    #define __float_vector __m512
    #define __double_vector __m512d
    #define __int_vector __mmask16
    #define FLOAT_VEC_SIZE 16
    #define DOUBLE_VEC_SIZE 8
    #define simd_malloc(size) (alligned_malloc(512, size))

    inline __attribute__((always_inline)) void _float_storeu(float* addr, const __float_vector A) {
        _mm512_storeu_ps(addr, A);
    }

    inline __attribute__((always_inline)) void _double_storeu(double* addr, const __double_vector A) {
        _mm512_storeu_pd(addr, A);
    }

    inline __attribute__((always_inline)) __float_vector _float_loadu(const float* addr) {
        return _mm512_loadu_ps(addr);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu(const double* addr) {
        return _mm512_loadu_pd(addr);
    }

    inline __attribute__((always_inline)) __int_vector _int_loadu(const void* addr) {
        return _load_mask16((__int_vector*) addr);
    }

    inline __attribute__((always_inline)) __float_vector _float_maskload(const float* addr, const __int_vector mask) {
        return _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, addr);
    }

    inline __attribute__((always_inline)) __double_vector _double_maskload(const double* addr, const __int_vector mask) {
        return _mm512_mask_loadu_pd(_mm512_setzero_pd(), (__mmask8) mask, addr);
    }

    inline __attribute__((always_inline)) void _float_maskstore(float* addr, const __int_vector mask, const __float_vector A) {
        _mm512_mask_storeu_ps(addr, mask, A);
    }

    inline __attribute__((always_inline)) void _double_maskstore(double* addr, const __int_vector mask, __double_vector A) {
        _mm512_mask_storeu_pd(addr, mask, A);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu2(const double* A, const double* B) {
        return _mm512_insertf64x4(_mm512_setzero_pd(), _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(A)), _mm_loadu_pd(B), 1), 0);
    }

    inline __attribute__((always_inline)) __double_vector _double_loadu2_from_float(const float* A, const float* B) {
        return _double_loadu((double[4]) {A[0], A[1], B[0], B[1]});
    }

    inline __attribute__((always_inline)) __float_vector _float_add_vec(__float_vector A, __float_vector B) {
        return _mm512_add_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_add_vec(__double_vector A, __double_vector B) {
        return _mm512_add_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_sub_vec(__float_vector A, __float_vector B) {
        return _mm512_sub_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_sub_vec(__double_vector A, __double_vector B) {
        return _mm512_sub_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mul_vec(__float_vector A, __float_vector B) {
        return _mm512_mul_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_mul_vec(__double_vector A, __double_vector B) {
        return _mm512_mul_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_div_vec(__float_vector A, __float_vector B) {
        return _mm512_div_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_div_vec(__double_vector A, __double_vector B) {
        return _mm512_div_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_set1_vec(float a) {
        return _mm512_set1_ps(a);
    }

    inline __attribute__((always_inline)) __double_vector _double_set1_vec(double a) {
        return _mm512_set1_pd(a);
    }

    inline __attribute__((always_inline)) __float_vector _float_set_vec(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l, float m, float n, float o, float p) {
        return _mm512_set_ps(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    }

    inline __attribute__((always_inline)) __double_vector _double_set_vec(double a, double b, double c, double d, double e, double f, double g, double h) {
        return _mm512_set_pd(a, b, c, d, e, f, g, h);
    }

    inline __attribute__((always_inline)) __float_vector _float_abs_vec(__float_vector x, __float_vector sign_mask) {
        return _mm512_andnot_ps(sign_mask, x);
    }

    inline __attribute__((always_inline)) __double_vector _double_abs_vec(__double_vector x, __double_vector sign_mask) {
        return _mm512_andnot_pd(sign_mask, x);
    }

    inline __attribute__((always_inline)) __float_vector _float_max_vec(__float_vector A, __float_vector B) {
        return _mm512_max_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_max_vec(__double_vector A, __double_vector B) {
        return _mm512_max_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_max_vec(__float_vector A, __float_vector B, __int_vector mask) {
        return _mm512_mask_max_ps(_mm512_set1_ps(-FLT_MAX), mask, A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_max_vec(__double_vector A, __double_vector B, __int_vector mask) {
        return _mm512_mask_max_ps(_mm512_set1_ps(-DBL_MAX), (__mmask8) mask, A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_min_vec(__float_vector A, __float_vector B) {
        return _mm512_min_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(__double_vector A, __double_vector B) {
        return _mm512_min_pd(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_min_vec(__float_vector A, __float_vector B, __int_vector mask) {
        return _mm512_mask_min_ps(_mm512_set1_ps(FLT_MAX), mask, A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_min_vec(__double_vector A, __double_vector B, __int_vector mask) {
        return _mm512_mask_min_ps(_mm512_set1_ps(DBL_MAX), (__mmask8) mask, A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_setzero_vec() {
        return _mm512_setzero_ps();
    }

    inline __attribute__((always_inline)) __double_vector _double_setzero_vec() {
        return _mm512_setzero_pd();
    }

    inline __attribute__((always_inline)) float _float_index_vec(const __float_vector A, const int i) {
        return A[i];
    }

    inline __attribute__((always_inline)) double _double_index_vec(const __double_vector A, const int i) {
        return A[i];
    }

    inline __attribute__((always_inline)) __float_vector _float_recp_vec(const __float_vector A) {
        return _mm512_rcp14_ps(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_recp_vec(const __double_vector A) {
        return _mm512_rcp14_pd(A);
    }

    inline __attribute__((always_inline)) __float_vector _float_rsqrt_vec(const __float_vector A) {
        return _mm512_rsqrt14_ps(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_rsqrt_vec(const __double_vector A) {
        return _mm512_rsqrt14_pd(A);
    }

    inline __attribute__((always_inline)) __float_vector _float_sqrt_vec(const __float_vector A) {
        return _mm512_rcp14_ps(_mm512_rsqrt14_ps(A));
    }

    inline __attribute__((always_inline)) __double_vector _double_sqrt_vec(const __double_vector A) {
        return _mm512_rcp14_pd(_mm512_rsqrt14_pd(A));
    }

#else
/** No SIMD Support **/
    #define __int_vector int
    #define __float_vector float
    #define __double_vector double
    #define FLOAT_VEC_SIZE 1
    #define DOUBLE_VEC_SIZE 1
    #define simd_malloc malloc

    inline __attribute__((always_inline)) void _float_storeu(float* addr, const __float_vector A) {
        addr[0] = A;
    }

    inline __attribute__((always_inline)) void _double_storeu(double* addr, const __double_vector A) {
        addr[0] = A;
    }

    inline __attribute__((always_inline))  __float_vector _float_loadu(const float* addr) {
        return addr[0];
    }

    inline __attribute__((always_inline))  __double_vector _double_loadu(const double* addr) {
        return addr[0];
    }

    inline __attribute__((always_inline)) __int_vector _int_loadu(const void* addr) {
        return ((int*) addr)[0];
    }

    inline __attribute__((always_inline)) __float_vector _float_maskload(const float* addr, const __int_vector mask) {
        return mask != 0 ? addr[0] : 0;
    }

    inline __attribute__((always_inline)) __double_vector _double_maskload(const double* addr, const __int_vector mask) {
        return mask != 0 ? addr[0] : 0;
    }

    inline __attribute__((always_inline)) void _float_maskstore(float* addr, const __int_vector mask, const __float_vector A) {
        mask != 0 ? addr[0] = A: 0;
    }

    inline __attribute__((always_inline)) void _double_maskstore(double* addr, const __int_vector mask, __double_vector A) {
        mask != 0 ? addr[0] = A: 0;
    }

    inline __attribute__((always_inline)) __float_vector _float_add_vec(const __float_vector A, const __float_vector B) {
        return A+B;
    }

    inline __attribute__((always_inline)) __double_vector _double_add_vec(const __double_vector A, const __double_vector B) {
        return A+B;
    }

    inline __attribute__((always_inline)) __float_vector _float_sub_vec(const __float_vector A, const __float_vector B) {
        return A-B;
    }

    inline __attribute__((always_inline)) __double_vector _double_sub_vec(const __double_vector A, const __double_vector B) {
        return A-B;
    }

    inline __attribute__((always_inline)) __float_vector _float_mul_vec(const __float_vector A, const __float_vector B) {
        return A*B;
    }

    inline __attribute__((always_inline)) __double_vector _double_mul_vec(const __double_vector A, const __double_vector B) {
        return A*B;
    }

    inline __attribute__((always_inline)) __float_vector _float_div_vec(const __float_vector A, const __float_vector B) {
        return A/B;
    }

    inline __attribute__((always_inline)) __double_vector _double_div_vec(const __double_vector A, const __double_vector B) {
        return A/B;
    }

    inline __attribute__((always_inline)) __float_vector _float_set1_vec(float a) {
        return a;
    }

    inline __attribute__((always_inline)) __double_vector _double_set1_vec(double a) {
        return a;
    }

    inline __attribute__((always_inline)) __float_vector _float_abs_vec(const __float_vector x, const __float_vector sign_mask) {
        return fabsf(x);
    }

    inline __attribute__((always_inline)) __double_vector _double_abs_vec(const __double_vector x, const __double_vector sign_mask) {
        return fabs(x);
    }

    inline __attribute__((always_inline)) __float_vector _float_max_vec(const __float_vector A, const __float_vector B) {
        return max(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_max_vec(const __double_vector A, const __double_vector B) {
        return max(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_max_vec(__float_vector A, __float_vector B, __int_vector mask) {
        return mask != 0 ? max(A, B) : -FLT_MAX;
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_max_vec(__double_vector A, __double_vector B, __int_vector mask) {
        return mask != 0 ? max(A, B) : -DBL_MAX;
    }

    inline __attribute__((always_inline)) __float_vector _float_min_vec(const __float_vector A,const  __float_vector B) {
        return min(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(const __double_vector A, const __double_vector B) {
        return min(A, B);
    }

    inline __attribute__((always_inline)) __float_vector _float_mask_min_vec(__float_vector A, __float_vector B, __int_vector mask) {
        return mask != 0 ? min(A, B) : FLT_MAX;
    }

    inline __attribute__((always_inline)) __double_vector _double_mask_min_vec(__double_vector A, __double_vector B, __int_vector mask) {
        return mask != 0 ? min(A, B) : DBL_MAX;
    }

    inline __attribute__((always_inline)) __float_vector _float_setzero_vec() {
        return 0;
    }

    inline __attribute__((always_inline)) __double_vector _double_setzero_vec() {
        return 0;
    }

    inline __attribute__((always_inline)) float _float_index_vec(const __float_vector A, const int i) {
        return A;
    }

    inline __attribute__((always_inline)) double _double_index_vec(const __double_vector A, const int i) {
        return A;
    }

    inline __attribute__((always_inline)) __float_vector _float_recp_vec(const __float_vector A) {
        return 1.f/A;
    }

    inline __attribute__((always_inline)) __double_vector _double_recp_vec(const __double_vector A) {
        return 1./A;
    }

    inline __attribute__((always_inline)) __float_vector _float_rsqrt_vec(const __float_vector A) {
        return 1.f/sqrtf(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_rsqrt_vec(const __double_vector A) {
        return 1./sqrt(A);
    }

    inline __attribute__((always_inline)) __float_vector _float_sqrt_vec(const __float_vector A) {
        return sqrtf(A);
    }

    inline __attribute__((always_inline)) __double_vector _double_sqrt_vec(const __double_vector A) {
        return sqrt(A);
    }
#endif