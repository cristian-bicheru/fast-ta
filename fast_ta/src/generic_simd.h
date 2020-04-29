#pragma once
#include <math.h>

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
    #define FLOAT_VEC_SIZE 8
    #define DOUBLE_VEC_SIZE 4
    #define simd_malloc(size) (alligned_malloc(256, size))

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

    inline __attribute__((always_inline)) __float_vector _float_min_vec(__float_vector A, __float_vector B) {
        return _mm256_min_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(__double_vector A, __double_vector B) {
        return _mm256_min_pd(A, B);
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

    inline __attribute__((always_inline)) __float_vector _float_min_vec(__float_vector A, __float_vector B) {
        return _mm_min_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(__double_vector A, __double_vector B) {
        return _mm_min_pd(A, B);
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

    inline __attribute__((always_inline)) __float_vector _float_min_vec(__float_vector A, __float_vector B) {
        return _mm512_min_ps(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(__double_vector A, __double_vector B) {
        return _mm512_min_pd(A, B);
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

    inline __attribute__((always_inline)) __float_vector _float_min_vec(const __float_vector A,const  __float_vector B) {
        return min(A, B);
    }

    inline __attribute__((always_inline)) __double_vector _double_min_vec(const __double_vector A, const __double_vector B) {
        return min(A, B);
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