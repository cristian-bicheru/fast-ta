#include <cstdlib>

#include "gtest/gtest.h"
#include "testing_common.h"

extern "C" {
    #include "volume_backend.h"
}

double ADI_REF_DOUBLE[data_len] = {0};
float ADI_REF_FLOAT[data_len] = {0};

double CMF_REF_DOUBLE[data_len] = {0};
float CMF_REF_FLOAT[data_len] = {0};

double EMV_REF_DOUBLE[data_len] = {0};
float EMV_REF_FLOAT[data_len] = {0};

double EMV_SMA_REF_DOUBLE[data_len] = {0};
float EMV_SMA_REF_FLOAT[data_len] = {0};

double FI_REF_DOUBLE[data_len] = {0};
float FI_REF_FLOAT[data_len] = {0};

double MFI_REF_DOUBLE[data_len] = {0};
float MFI_REF_FLOAT[data_len] = {0};

double NVI_REF_DOUBLE[data_len] = {0};
float NVI_REF_FLOAT[data_len] = {0};

double OBV_REF_DOUBLE[data_len] = {0};
float OBV_REF_FLOAT[data_len] = {0};

double VPT_REF_DOUBLE[data_len] = {0};
float VPT_REF_FLOAT[data_len] = {0};

double VWAP_REF_DOUBLE[data_len] = {0};
float VWAP_REF_FLOAT[data_len] = {0};

TEST(momentum_backend, ADIDouble) {
    double* out = _ADI_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(ADI_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, ADIFloat) {
    float* out = _ADI_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len);
    double max_fp_error = get_max_fp_error(ADI_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(ADI_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, CMFDouble) {
    double* out = _CMF_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len, 20);
    for (int i=0; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(CMF_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, CMFFloat) {
    float* out = _CMF_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len, 20);
    double max_fp_error = get_max_fp_error(CMF_REF_DOUBLE, data_len);
    for (int i=0; i<data_len; i++) {
        ASSERT_NEAR(CMF_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, EMVDouble) {
    double** out = _EMV_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len, 14);
    for (int i=1; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(EMV_REF_DOUBLE[i], out[0][i]);
        ASSERT_DOUBLE_EQ(EMV_SMA_REF_DOUBLE[i], out[1][i]);
    }

    free(out[0]);
    free(out);
}

TEST(momentum_backend, EMVFloat) {
    float** out = _EMV_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_VOLUME_FLOAT, data_len, 14);
    double max_fp_error1 = get_max_fp_error(EMV_REF_DOUBLE, data_len);
    double max_fp_error2 = get_max_fp_error(EMV_SMA_REF_DOUBLE, data_len);
    for (int i=1; i<data_len; i++) {
        ASSERT_NEAR(EMV_REF_FLOAT[i], out[0][i], max_fp_error1);
        ASSERT_NEAR(EMV_SMA_REF_FLOAT[i], out[1][i], max_fp_error2);
    }

    free(out[0]);
    free(out);
}

TEST(momentum_backend, FIDouble) {
    double* out = _FI_DOUBLE(SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len, 13);
    for (int i=1; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(FI_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, FIFloat) {
    float* out = _FI_FLOAT(SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len, 13);
    double max_fp_error = get_max_fp_error(FI_REF_DOUBLE, data_len);
    for (int i=1; i<data_len; i++) {
        ASSERT_NEAR(FI_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, MFIDouble) {
    double* out = _MFI_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len, 14);
    for (int i=14; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(MFI_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, MFIFloat) {
    float* out = _MFI_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len, 14);
    double max_fp_error = get_max_fp_error(MFI_REF_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_NEAR(MFI_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, NVIDouble) {
    double* out = _NVI_DOUBLE(SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(NVI_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, NVIFloat) {
    float* out = _NVI_FLOAT(SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len);
    double max_fp_error = get_max_fp_error(MFI_REF_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_NEAR(NVI_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, OBVDouble) {
    double* out = _OBV_DOUBLE(SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(OBV_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, OBVFloat) {
    float* out = _OBV_FLOAT(SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len);
    double max_fp_error = get_max_fp_error(OBV_REF_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_NEAR(OBV_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, VPTDouble) {
    double* out = _VPT_DOUBLE(SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(VPT_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, VPTFloat) {
    float* out = _VPT_FLOAT(SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len);
    double max_fp_error = get_max_fp_error(VPT_REF_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_NEAR(VPT_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

TEST(momentum_backend, VWAPDouble) {
    double* out = _VWAP_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len, 14);
    for (int i=14; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(VWAP_REF_DOUBLE[i], out[i]);
    }

    free(out);
}

TEST(momentum_backend, VWAPFloat) {
    float* out = _VWAP_FLOAT(SAMPLE_HIGH_FLOAT, SAMPLE_LOW_FLOAT, SAMPLE_CLOSE_FLOAT, SAMPLE_VOLUME_FLOAT, data_len, 14);
    double max_fp_error = get_max_fp_error(VWAP_REF_DOUBLE, data_len);
    for (int i=14; i<data_len; i++) {
        ASSERT_NEAR(VWAP_REF_FLOAT[i], out[i], max_fp_error);
    }

    free(out);
}

int main(int argc, char* argv[]) {
    populate_float_arrays();

    load_data_fours("fast_ta/src/test_data/adi_ref.txt", ADI_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/cmf_ref.txt", CMF_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/emv_ref.txt", EMV_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/emv_sma_ref.txt", EMV_SMA_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/fi_ref.txt", FI_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/mfi_ref.txt", MFI_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/nvi_ref.txt", NVI_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/obv_ref.txt", OBV_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/vpt_ref.txt", VPT_REF_DOUBLE);
    load_data_fours("fast_ta/src/test_data/vwap_ref.txt", VWAP_REF_DOUBLE);

    for (int i=0; i<data_len; i++) {
        ADI_REF_FLOAT[i] = (float)ADI_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        CMF_REF_FLOAT[i] = (float)CMF_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        EMV_REF_FLOAT[i] = (float)EMV_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        EMV_SMA_REF_FLOAT[i] = (float)EMV_SMA_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        FI_REF_FLOAT[i] = (float)FI_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        MFI_REF_FLOAT[i] = (float)MFI_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        NVI_REF_FLOAT[i] = (float)NVI_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        OBV_REF_FLOAT[i] = (float)OBV_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        VPT_REF_FLOAT[i] = (float)VPT_REF_DOUBLE[i];
    }
    for (int i=0; i<data_len; i++) {
        VWAP_REF_FLOAT[i] = (float)VWAP_REF_DOUBLE[i];
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}