#include <iostream>
#include <stdlib.h>
#include <iomanip>

#include "fast_ta/src/testing_common.h"
#include "fast_ta/src/2darray.h"

extern "C" {
    #include "fast_ta/src/momentum_backend.h"
    #include "fast_ta/src/volume_backend.h"
    #include "fast_ta/src/volatility_backend.h"
}

void pprint(double* arr) {
    int column_width = 4;
    std::cout.precision(16);
    for (int i=0; i<data_len; i++) {
        std::cout << std::setw(18) << arr[i] << ",";
        if ((i+1)%column_width == 0 || i == data_len-1) {
            std::cout << "\n";
        }
    }
}

int main() {
    // your indicator here
    double* out = _WILLIAMS_R_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_CLOSE_DOUBLE, 14, data_len);
    pprint(out);
    free(out);
}
