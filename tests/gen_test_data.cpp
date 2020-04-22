#include <iostream>
#include <stdlib.h>
#include <iomanip>

#include "fast_ta/src/testing_common.h"

extern "C" {
    #include "fast_ta/src/momentum_backend.h"
    #include "fast_ta/src/volume_backend.h"
}


int main() {
    int window_size = 14;
    // your indicator here
    double* out = _ADI_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len);

    int column_width = 4;
    std::cout.precision(16);
    for (int i=0; i<data_len; i++) {
        std::cout << std::setw(18) << out[i] << ",";
        if ((i+1)%column_width == 0 || i == data_len-1) {
            std::cout << "\n";
        }
    }

    free(out);
}
