#include <iostream>
#include <stdlib.h>
#include <iomanip>

#include "fast_ta/src/testing_common.h"
#include "fast_ta/src/array_pair.h"

extern "C" {
    #include "fast_ta/src/momentum_backend.h"
    #include "fast_ta/src/volume_backend.h"
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
    int window_size = 14;
    // your indicator here
    struct double_array_pair out = _EMV_DOUBLE(SAMPLE_HIGH_DOUBLE, SAMPLE_LOW_DOUBLE, SAMPLE_VOLUME_DOUBLE, data_len, 14);
    pprint(out.arr1);
    pprint(out.arr2);
    free(out.arr1);
    free(out.arr2);
}
