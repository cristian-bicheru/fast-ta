#include <cstdlib>

#include "gtest/gtest.h"
#include "testing_common.h"

extern "C" {
    #include "volatility_backend.h"
}

double ATR_REF_DOUBLE[data_len] = {0};
float ATR_REF_FLOAT[data_len] = {0};

double BOLL_REF_DOUBLE[data_len] = {0};
float BOLL_REF_FLOAT[data_len] = {0};

double BOLM_REF_DOUBLE[data_len] = {0};
float BOLM_REF_FLOAT[data_len] = {0};

double BOLH_REF_DOUBLE[data_len] = {0};
float BOLH_REF_FLOAT[data_len] = {0};

TEST(momentum_backend, ATRDouble) {
    double* out = _ATR_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, data_len, 14);
    for (int i=0; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(ATR_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, ATRFloat) {
    float* out = _ATR_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, data_len, 14);
    double max_fp_error = get_max_fp_error(ATR_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(ATR_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, BOLDouble) {
    double** out = _BOL_DOUBLE(SAMPLE_CLOSE_DOUBLE, data_len, 20, 2);
    for (int i=0; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(BOLL_REF_DOUBLE[i], out[0][i]);
        ASSERT_DOUBLE_EQ(BOLM_REF_DOUBLE[i], out[1][i]);
        ASSERT_DOUBLE_EQ(BOLH_REF_DOUBLE[i], out[2][i]);
    }
    free(out[0]);
    free(out);
}

TEST(momentum_backend, BOLFloat) {
    float** out = _BOL_FLOAT(SAMPLE_CLOSE_FLOAT, data_len, 20, 2);
    double max_fp_error1 = get_max_fp_error(BOLL_REF_DOUBLE, data_len);
    double max_fp_error2 = get_max_fp_error(BOLM_REF_DOUBLE, data_len);
    double max_fp_error3 = get_max_fp_error(BOLH_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(BOLL_REF_FLOAT[i], out[0][i], max_fp_error1);
        ASSERT_NEAR(BOLM_REF_FLOAT[i], out[1][i], max_fp_error2);
        ASSERT_NEAR(BOLH_REF_FLOAT[i], out[2][i], max_fp_error3);
    }

    free(out[0]);
    free(out);
}

int main(int argc, char* argv[]) {
    populate_float_arrays();

    load_data_fours("fast_ta/src/test_data/atr_ref.txt", ATR_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/boll_ref.txt", BOLL_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/bolm_ref.txt", BOLM_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/bolh_ref.txt", BOLH_REF_DOUBLE);

    for (int i=0; i<data_len; i++) {
        ATR_REF_FLOAT[i] = (float)ATR_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        BOLL_REF_FLOAT[i] = (float)BOLL_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        BOLM_REF_FLOAT[i] = (float)BOLM_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        BOLH_REF_FLOAT[i] = (float)BOLH_REF_DOUBLE[i];
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}