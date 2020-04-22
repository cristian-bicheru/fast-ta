#include <iostream>
#include <stdlib.h>
#include <iomanip>

#include "testing_common.h"

extern "C" {
    #include "momentum_backend.h"
}


int main() {
    int window_size = 14;
    // your indicator here
    double* out = _RSI_DOUBLE(SAMPLE_CLOSE_DOUBLE, NULL, data_len, window_size, 0);

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
