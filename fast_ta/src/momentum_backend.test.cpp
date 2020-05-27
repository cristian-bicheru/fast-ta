#include <cstdlib>

#include "gtest/gtest.h"
#include "testing_common.h"

extern "C" {
    #include "momentum/momentum_backend.h"
}

double RSI_REF_DOUBLE[data_len] = {0};
float RSI_REF_FLOAT[data_len] = {0};

double AO_REF_DOUBLE[data_len] = {0};
float AO_REF_FLOAT[data_len] = {0};

double KAMA_REF_DOUBLE[data_len] = {0};
float KAMA_REF_FLOAT[data_len] = {0};

double ROC_REF_DOUBLE[data_len] = {0};
float ROC_REF_FLOAT[data_len] = {0};

double STOCH_REF_DOUBLE[data_len] = {0};
float STOCH_REF_FLOAT[data_len] = {0};

double STOCH_SIGNAL_REF_DOUBLE[data_len] = {0};
float STOCH_SIGNAL_REF_FLOAT[data_len] = {0};

double TSI_REF_DOUBLE[data_len] = {0};
float TSI_REF_FLOAT[data_len] = {0};

double UO_REF_DOUBLE[data_len] = {0};
float UO_REF_FLOAT[data_len] = {0};

double WR_REF_DOUBLE[data_len] = {0};
float WR_REF_FLOAT[data_len] = {0};

TEST(momentum_backend, RSIDouble) {
    int window_size = 14;
    double max_dp_error = get_max_dp_error(RSI_REF_DOUBLE, data_len);
    double* out = _RSI_DOUBLE(SAMPLE_CLOSE_DOUBLE, data_len, window_size);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(RSI_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, RSIFloat) {
    int window_size = 14;

    float* out = _RSI_FLOAT(SAMPLE_CLOSE_FLOAT, data_len, window_size);
    double max_fp_error = get_max_fp_error(RSI_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(RSI_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, AODouble) {
    double* out =  _AO_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, 5, 34, data_len);
    double max_dp_error = get_max_dp_error(AO_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(AO_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, AOFloat) {
    float* out = _AO_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, 5, 34, data_len);
    double max_fp_error = get_max_fp_error(AO_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(AO_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, KAMADouble) {
    double* out =  _KAMA_DOUBLE(SAMPLE_CLOSE_DOUBLE, 10, 2, 30, data_len);
    double max_dp_error = get_max_dp_error(KAMA_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(KAMA_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, KAMAFloat) {
    float* out = _KAMA_FLOAT(SAMPLE_CLOSE_FLOAT, 10, 2, 30, data_len);
    double max_fp_error = get_max_fp_error(KAMA_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(KAMA_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, ROCDouble) {
    double* out = _ROC_DOUBLE(SAMPLE_CLOSE_DOUBLE, 12, data_len);
    double max_dp_error = get_max_dp_error(ROC_REF_DOUBLE, data_len);
    for (int i=12; i<data_len; i++) {
        ASSERT_NEAR(ROC_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, ROCFloat) {
    float* out;
    for (int i = 0; i < 1000000; i++) {
        out = _ROC_FLOAT(SAMPLE_CLOSE_FLOAT, 12, data_len);
        free(out);
    }
    out =  _ROC_FLOAT(SAMPLE_CLOSE_FLOAT, 12, data_len);
    double max_fp_error = get_max_fp_error(ROC_REF_DOUBLE, data_len);
    for (int i=12; i<data_len; i++) {
        ASSERT_NEAR(ROC_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, STOCHDouble) {
    double* out = _STOCHASTIC_OSCILLATOR_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, 14, 3, data_len);
    double max_dp_error1 = get_max_dp_error(STOCH_REF_DOUBLE, data_len);
    double max_dp_error2 = get_max_dp_error(STOCH_SIGNAL_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(STOCH_REF_DOUBLE[i], out[i], max_dp_error1);
        ASSERT_NEAR(STOCH_SIGNAL_REF_DOUBLE[i], out[i+data_len], max_dp_error2);
    }

    free(out);
}

TEST(momentum_backend, STOCHFloat) {
    float* out =  _STOCHASTIC_OSCILLATOR_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, 14, 3, data_len);
    double max_fp_error1 = get_max_fp_error(STOCH_REF_DOUBLE, data_len);
    double max_fp_error2 = get_max_fp_error(STOCH_SIGNAL_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(STOCH_REF_FLOAT[i], out[i], max_fp_error1);
        ASSERT_NEAR(STOCH_SIGNAL_REF_FLOAT[i], out[i+data_len], max_fp_error2);
    }

    free(out);
}

TEST(momentum_backend, TSIDouble) {
    double* out = _TSI_DOUBLE(SAMPLE_CLOSE_DOUBLE, 25, 13, data_len);
    double max_dp_error = get_max_dp_error(KAMA_REF_DOUBLE, data_len);
    for (int i=1; i<data_len; i++) {
        ASSERT_NEAR(TSI_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, TSIFloat) {
    float* out =  _TSI_FLOAT(SAMPLE_CLOSE_FLOAT, 25, 13, data_len);
    double max_fp_error = get_max_fp_error(TSI_REF_DOUBLE, data_len);
    for (int i=1; i<data_len; i++) {
        ASSERT_NEAR(TSI_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, UODouble) {
    double* out = _ULTIMATE_OSCILLATOR_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, 7, 14, 28, 4, 2, 1, data_len);
    double max_dp_error = get_max_dp_error(UO_REF_DOUBLE, data_len);
    for (int i=28; i<data_len; i++) {
        ASSERT_NEAR(UO_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, UOFloat) {
    float* out =  _ULTIMATE_OSCILLATOR_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, 7, 14, 28, 4, 2, 1, data_len);
    double max_fp_error = get_max_fp_error(UO_REF_DOUBLE, data_len);
    for (int i=28; i<data_len; i++) {
        ASSERT_NEAR(UO_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, WRDouble) {
    double* out = _WILLIAMS_R_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, 14, data_len);
    double max_dp_error = get_max_dp_error(WR_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(WR_REF_DOUBLE[i], out[i], max_dp_error);
    }

    free(out);
}

TEST(momentum_backend, WRFloat) {
    float* out =  _WILLIAMS_R_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, 14, data_len);
    double max_fp_error = get_max_fp_error(WR_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(WR_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

int main(int argc, char* argv[]) {
    populate_float_arrays();

    load_data_fours("fast_ta/src/test_data/rsi_ref.txt", RSI_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/ao_ref.txt", AO_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/kama_ref.txt", KAMA_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/roc_ref.txt", ROC_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/stoch_ref.txt", STOCH_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/stoch_signal_ref.txt",
                    STOCH_SIGNAL_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/tsi_ref.txt", TSI_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/uo_ref.txt", UO_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/wr_ref.txt", WR_REF_DOUBLE);

    for (int i=0; i<data_len; i++) {
        RSI_REF_FLOAT[i] = (float)RSI_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        AO_REF_FLOAT[i] = (float)AO_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        KAMA_REF_FLOAT[i] = (float)KAMA_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        ROC_REF_FLOAT[i] = (float)ROC_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        STOCH_REF_FLOAT[i] = (float)STOCH_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        STOCH_SIGNAL_REF_FLOAT[i] = (float)STOCH_SIGNAL_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        TSI_REF_FLOAT[i] = (float)TSI_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        UO_REF_FLOAT[i] = (float)UO_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        WR_REF_FLOAT[i] = (float)WR_REF_DOUBLE[i];
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
