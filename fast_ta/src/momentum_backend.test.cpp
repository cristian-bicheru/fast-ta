#include "gtest/gtest.h"

#include <stdlib.h>

extern "C" {
    #include "momentum_backend.h"
}

const int data_len = 253;
const double SAMPLE_OPEN[data_len] = {
    199.250000, 203.130005, 203.860001, 204.529999, 207.479996, 207.160004,
    205.279999, 204.300003, 204.610001, 200.669998, 210.520004, 209.149994,
    211.750000, 208.479996, 202.860001, 202.899994, 200.720001, 197.179993,
    185.720001, 188.660004, 190.919998, 190.080002, 189.000000, 183.089996,
    186.600006, 182.779999, 179.660004, 178.970001, 178.229996, 177.380005,
    178.300003, 175.070007, 173.300003, 179.639999, 182.539993, 185.220001,
    190.149994, 192.580002, 194.809998, 194.190002, 194.149994, 192.740005,
    193.889999, 198.449997, 197.869995, 199.460007, 198.779999, 198.580002,
    195.570007, 199.800003, 199.740005, 197.919998, 201.550003, 202.729996,
    204.410004, 204.229996, 200.020004, 201.240005, 203.229996, 201.750000,
    203.300003, 205.210007, 204.500000, 203.350006, 205.660004, 202.589996,
    207.220001, 208.839996, 208.669998, 207.020004, 207.740005, 209.679993,
    208.779999, 213.039993, 208.429993, 204.020004, 193.339996, 197.000000,
    199.039993, 203.429993, 200.990005, 200.479996, 208.970001, 202.750000,
    201.740005, 206.500000, 210.350006, 210.360001, 212.639999, 212.460007,
    202.639999, 206.490005, 204.160004, 205.529999, 209.009995, 208.740005,
    205.699997, 209.190002, 213.279999, 213.259995, 214.169998, 216.699997,
    223.589996, 223.089996, 218.750000, 219.899994, 220.699997, 222.770004,
    220.960007, 217.729996, 218.720001, 217.679993, 221.029999, 219.889999,
    218.820007, 223.970001, 224.589996, 218.960007, 220.820007, 227.009995,
    227.059998, 224.399994, 227.029999, 230.089996, 236.210007, 235.869995,
    235.320007, 234.369995, 235.279999, 236.410004, 240.509995, 239.960007,
    243.179993, 243.580002, 246.580002, 249.050003, 243.289993, 243.259995,
    248.759995, 255.820007, 257.500000, 257.130005, 257.239990, 259.429993,
    260.140015, 262.200012, 261.959991, 264.470001, 262.640015, 265.760010,
    267.100006, 266.290009, 263.190002, 262.010010, 261.779999, 266.369995,
    264.290009, 267.839996, 267.250000, 264.160004, 259.450012, 261.739990,
    265.579987, 270.709991, 266.920013, 268.480011, 270.769989, 271.459991,
    275.149994, 279.859985, 280.410004, 279.739990, 280.019989, 279.440002,
    284.000000, 284.269989, 289.910004, 289.799988, 291.519989, 293.649994,
    300.350006, 297.429993, 299.799988, 298.390015, 303.190002, 309.630005,
    310.329987, 316.959991, 312.679993, 311.339996, 315.239990, 318.730011,
    316.570007, 317.700012, 319.230011, 318.309998, 308.950012, 317.690002,
    324.339996, 323.869995, 309.510010, 308.660004, 318.850006, 321.450012,
    325.209991, 320.029999, 321.549988, 319.609985, 327.200012, 324.869995,
    324.950012, 319.000000, 323.619995, 320.299988, 313.049988, 298.179993,
    288.079987, 292.649994, 273.519989, 273.359985, 298.809998, 289.320007,
    302.739990, 292.920013, 289.029999, 266.170013, 285.339996, 275.429993,
    248.229996, 277.970001, 242.210007, 252.860001, 246.669998, 244.779999,
    229.240005, 224.369995, 246.880005, 245.520004, 258.440002, 247.740005,
    254.809998, 254.289993, 240.910004, 244.929993, 241.410004, 262.470001,
    259.429993, 266.070007, 267.989990, 273.250000, 287.049988, 284.429993,
    286.559998
};

const double SAMPLE_CLOSE[data_len] = {
    199.250000, 203.130005, 203.860001, 204.529999, 207.479996, 207.160004,
    205.279999, 204.300003, 204.610001, 200.669998, 210.520004, 209.149994,
    211.750000, 208.479996, 202.860001, 202.899994, 200.720001, 197.179993,
    185.720001, 188.660004, 190.919998, 190.080002, 189.000000, 183.089996,
    186.600006, 182.779999, 179.660004, 178.970001, 178.229996, 177.380005,
    178.300003, 175.070007, 173.300003, 179.639999, 182.539993, 185.220001,
    190.149994, 192.580002, 194.809998, 194.190002, 194.149994, 192.740005,
    193.889999, 198.449997, 197.869995, 199.460007, 198.779999, 198.580002,
    195.570007, 199.800003, 199.740005, 197.919998, 201.550003, 202.729996,
    204.410004, 204.229996, 200.020004, 201.240005, 203.229996, 201.750000,
    203.300003, 205.210007, 204.500000, 203.350006, 205.660004, 202.589996,
    207.220001, 208.839996, 208.669998, 207.020004, 207.740005, 209.679993,
    208.779999, 213.039993, 208.429993, 204.020004, 193.339996, 197.000000,
    199.039993, 203.429993, 200.990005, 200.479996, 208.970001, 202.750000,
    201.740005, 206.500000, 210.350006, 210.360001, 212.639999, 212.460007,
    202.639999, 206.490005, 204.160004, 205.529999, 209.009995, 208.740005,
    205.699997, 209.190002, 213.279999, 213.259995, 214.169998, 216.699997,
    223.589996, 223.089996, 218.750000, 219.899994, 220.699997, 222.770004,
    220.960007, 217.729996, 218.720001, 217.679993, 221.029999, 219.889999,
    218.820007, 223.970001, 224.589996, 218.960007, 220.820007, 227.009995,
    227.059998, 224.399994, 227.029999, 230.089996, 236.210007, 235.869995,
    235.320007, 234.369995, 235.279999, 236.410004, 240.509995, 239.960007,
    243.179993, 243.580002, 246.580002, 249.050003, 243.289993, 243.259995,
    248.759995, 255.820007, 257.500000, 257.130005, 257.239990, 259.429993,
    260.140015, 262.200012, 261.959991, 264.470001, 262.640015, 265.760010,
    267.100006, 266.290009, 263.190002, 262.010010, 261.779999, 266.369995,
    264.290009, 267.839996, 267.250000, 264.160004, 259.450012, 261.739990,
    265.579987, 270.709991, 266.920013, 268.480011, 270.769989, 271.459991,
    275.149994, 279.859985, 280.410004, 279.739990, 280.019989, 279.440002,
    284.000000, 284.269989, 289.910004, 289.799988, 291.519989, 293.649994,
    300.350006, 297.429993, 299.799988, 298.390015, 303.190002, 309.630005,
    310.329987, 316.959991, 312.679993, 311.339996, 315.239990, 318.730011,
    316.570007, 317.700012, 319.230011, 318.309998, 308.950012, 317.690002,
    324.339996, 323.869995, 309.510010, 308.660004, 318.850006, 321.450012,
    325.209991, 320.029999, 321.549988, 319.609985, 327.200012, 324.869995,
    324.950012, 319.000000, 323.619995, 320.299988, 313.049988, 298.179993,
    288.079987, 292.649994, 273.519989, 273.359985, 298.809998, 289.320007,
    302.739990, 292.920013, 289.029999, 266.170013, 285.339996, 275.429993,
    248.229996, 277.970001, 242.210007, 252.860001, 246.669998, 244.779999,
    229.240005, 224.369995, 246.880005, 245.520004, 258.440002, 247.740005,
    254.809998, 254.289993, 240.910004, 244.929993, 241.410004, 262.470001,
    259.429993, 266.070007, 267.989990, 273.250000, 287.049988, 284.429993,
    286.559998
};

const double SAMPLE_HIGH[data_len] = {
    201.369995, 203.380005, 204.149994, 204.940002, 207.750000, 208.479996,
    207.759995, 205.000000, 205.970001, 203.399994, 215.309998, 212.649994,
    211.839996, 208.839996, 207.419998, 205.339996, 201.679993, 198.850006,
    189.479996, 189.699997, 191.750000, 192.470001, 190.899994, 184.350006,
    188.000000, 185.710007, 180.539993, 182.139999, 180.589996, 179.350006,
    179.229996, 177.990005, 177.919998, 179.830002, 184.990005, 185.470001,
    191.919998, 195.369995, 196.000000, 195.970001, 196.789993, 193.589996,
    194.960007, 200.289993, 199.880005, 200.610001, 200.850006, 200.160004,
    199.259995, 200.990005, 201.570007, 199.500000, 204.490005, 203.130005,
    204.440002, 205.080002, 201.399994, 201.509995, 203.729996, 204.389999,
    204.000000, 205.869995, 206.110001, 205.089996, 205.880005, 206.500000,
    207.229996, 208.910004, 209.149994, 209.240005, 209.729996, 210.639999,
    210.160004, 221.369995, 218.029999, 206.429993, 198.649994, 198.070007,
    199.559998, 203.529999, 202.759995, 202.050003, 212.139999, 206.440002,
    205.139999, 207.160004, 212.729996, 213.350006, 213.649994, 214.440002,
    212.050003, 207.190002, 208.550003, 205.720001, 209.320007, 210.449997,
    206.979996, 209.479996, 213.970001, 214.419998, 216.440002, 216.779999,
    223.710007, 226.419998, 220.789993, 220.130005, 220.820007, 222.850006,
    223.759995, 222.559998, 219.839996, 222.490005, 221.500000, 220.940002,
    220.960007, 224.580002, 228.220001, 223.580002, 220.960007, 227.490005,
    229.929993, 228.059998, 227.789993, 230.440002, 237.639999, 238.130005,
    237.649994, 235.240005, 236.149994, 237.580002, 240.990005, 242.199997,
    243.240005, 244.800003, 246.729996, 249.250000, 249.750000, 245.300003,
    249.169998, 255.929993, 257.850006, 258.190002, 257.489990, 260.350006,
    260.440002, 262.470001, 262.790009, 264.779999, 264.880005, 265.779999,
    267.429993, 268.000000, 266.079987, 264.010010, 263.179993, 266.440002,
    267.160004, 267.980011, 268.000000, 268.250000, 259.529999, 263.309998,
    265.890015, 271.000000, 270.799988, 270.070007, 271.100006, 272.559998,
    275.299988, 280.790009, 281.769989, 281.899994, 281.179993, 282.649994,
    284.250000, 284.890015, 289.980011, 293.970001, 292.690002, 293.679993,
    300.600006, 300.579987, 299.959991, 300.899994, 304.440002, 310.429993,
    312.670013, 317.070007, 317.570007, 315.500000, 315.700012, 318.739990,
    319.019989, 319.989990, 319.559998, 323.329987, 311.769989, 318.399994,
    327.850006, 324.089996, 322.679993, 313.489990, 319.640015, 324.760010,
    325.220001, 323.399994, 321.549988, 323.899994, 327.220001, 326.220001,
    325.980011, 319.750000, 324.570007, 324.649994, 320.450012, 304.179993,
    302.529999, 297.880005, 286.000000, 278.410004, 301.440002, 304.000000,
    303.399994, 299.549988, 290.820007, 278.089996, 286.440002, 281.220001,
    270.000000, 279.920013, 259.079987, 257.609985, 250.000000, 252.839996,
    251.830002, 228.500000, 247.690002, 258.250000, 258.679993, 255.869995,
    255.520004, 262.489990, 248.720001, 245.149994, 245.699997, 263.109985,
    271.700012, 267.369995, 270.070007, 273.700012, 288.250000, 286.329987,
    288.179993
};

const double SAMPLE_LOW[data_len] = {
    198.559998, 198.610001, 202.520004, 202.339996, 203.899994, 207.050003,
    205.119995, 202.119995, 203.860001, 199.110001, 209.229996, 208.130005,
    210.229996, 203.500000, 200.830002, 201.750000, 196.660004, 192.770004,
    182.850006, 185.410004, 186.020004, 188.839996, 186.759995, 180.279999,
    184.699997, 182.550003, 177.809998, 178.619995, 177.910004, 176.000000,
    176.669998, 174.990005, 170.270004, 174.520004, 181.139999, 182.149994,
    185.770004, 191.619995, 193.600006, 193.389999, 193.600006, 190.300003,
    192.169998, 195.210007, 197.309998, 198.029999, 198.149994, 198.169998,
    195.289993, 197.350006, 199.570007, 197.050003, 200.649994, 201.360001,
    202.690002, 202.899994, 198.410004, 198.809998, 201.559998, 201.710007,
    202.199997, 204.000000, 203.500000, 203.270004, 203.699997, 202.360001,
    203.610001, 207.289993, 207.169998, 206.729996, 207.139999, 208.440002,
    207.309998, 211.300003, 206.740005, 201.630005, 192.580002, 194.039993,
    193.820007, 199.389999, 199.289993, 199.149994, 200.479996, 202.589996,
    199.669998, 203.839996, 210.029999, 210.320007, 211.600006, 210.750000,
    201.000000, 205.059998, 203.529999, 203.320007, 206.660004, 207.199997,
    204.220001, 207.320007, 211.509995, 212.509995, 211.070007, 211.710007,
    217.729996, 222.860001, 217.020004, 217.559998, 219.119995, 219.440002,
    220.369995, 217.470001, 217.649994, 217.190002, 217.139999, 218.830002,
    217.279999, 220.789993, 224.199997, 217.929993, 215.130005, 223.889999,
    225.839996, 224.330002, 225.639999, 227.300003, 232.309998, 234.669998,
    234.880005, 233.199997, 233.520004, 234.289993, 237.320007, 239.619995,
    241.220001, 241.809998, 242.880005, 246.720001, 242.570007, 241.210007,
    237.259995, 249.160004, 255.380005, 256.320007, 255.369995, 258.109985,
    256.850006, 258.279999, 260.920013, 261.070007, 262.100006, 263.010010,
    264.230011, 265.390015, 260.399994, 261.179993, 260.839996, 262.519989,
    262.500000, 265.309998, 265.899994, 263.450012, 256.290009, 260.679993,
    262.730011, 267.299988, 264.910004, 265.859985, 268.500000, 267.320007,
    270.929993, 276.980011, 278.799988, 279.119995, 278.950012, 278.559998,
    280.369995, 282.920013, 284.700012, 288.119995, 285.220001, 289.519989,
    295.190002, 296.500000, 292.750000, 297.480011, 297.160004, 306.200012,
    308.250000, 311.149994, 312.170013, 309.549988, 312.089996, 315.000000,
    316.000000, 317.309998, 315.649994, 317.519989, 304.880005, 312.190002,
    321.380005, 318.750000, 308.290009, 302.220001, 313.630005, 318.950012,
    320.260010, 318.000000, 313.850006, 318.709991, 321.470001, 323.350006,
    322.850006, 314.609985, 320.000000, 318.209991, 310.500000, 289.230011,
    286.130005, 286.500000, 272.959991, 256.369995, 277.720001, 285.799988,
    293.130005, 291.410004, 281.230011, 263.000000, 269.369995, 271.859985,
    248.000000, 252.949997, 240.000000, 238.399994, 237.119995, 242.610001,
    228.000000, 212.610001, 234.300003, 244.300003, 246.360001, 247.050003,
    249.399994, 252.000000, 239.130005, 236.899994, 238.970001, 249.380005,
    259.000000, 261.230011, 264.700012, 265.829987, 278.049988, 280.630005,
    282.350189
};

const double RSI_REF[data_len] = {
    100.00000000000000, 100.00000000000000, 100.00000000000000, 100.00000000000000,
    100.00000000000000,  96.25739825599740,  78.90701364804372,  72.12974526092886,
     72.86692382849918,  54.53384069849339,  72.08938801872016,  68.41516330018673,
     71.20080548135459,  64.09159006212391,  54.70419594093941,  54.75498234981233,
     51.37366538534326,  46.36641211482944,  34.60695750293817,  38.88900229412889,
     42.03136925549301,  41.18371707643900,  40.06494724963638,  34.53608021003761,
     39.84547629767282,  36.38667233626698,  33.80554577706820,  33.24390229753503,
     32.61801143878002,  31.87565899924566,  33.63618025817780,  30.64213873734218,
     29.11283886972829,  40.55677353826410,  44.93572853682879,  48.69699891218202,
     54.81182802105524,  57.50069586037796,  59.86112487927482,  58.88194881592243,
     58.81509174730248,  56.38521651546807,  57.91247693282591,  63.38724302979215,
     62.27757413753249,  64.13137009502294,  62.71195007562881,  62.27538302190997,
     55.96123290980827,  61.81984856469659,  61.69448028508939,  57.86122021308213,
     62.82277502180742,  64.29450386330764,  66.33770514874277,  65.90255068271109,
     56.55813353629519,  58.39898001951338,  61.28107318893723,  58.05947582937085,
     60.40705060107074,  63.14466551527125,  61.44389286427465,  58.68670140963144,
     62.34217880278046,  55.33466298934897,  62.23000313985521,  64.30631852053867,
     63.90928683735919,  60.03493716734896,  61.14194655141731,  64.03281222235182,
     61.73811449069662,  67.64798888082808,  57.32848197634250,  49.54265734233129,
     36.58435005089367,  42.16698788881757,  45.06959089653871,  50.79305257539195,
     47.81121760716258,  47.18765152279995,  57.19580216912424,  49.75645276695353,
     48.64991475479461,  53.85807772956984,  57.60359673011219,  57.61321623813230,
     59.85108205547751,  59.58363734672355,  47.19324257055008,  51.45543446554826,
     48.88392036198381,  50.45187132398637,  54.28764504238800,  53.93876163817617,
     50.03931988246719,  54.13840232052485,  58.44166694861575,  58.41279770397405,
     59.39544627962980,  62.07825117529327,  68.23375638857365,  67.37901344763657,
     60.31625113063757,  61.46878468806555,  62.28929910350026,  64.40166007985925,
     61.17489401189029,  55.80174337509030,  57.04702369054166,  55.28483670110354,
     59.61255485187959,  57.57058765907507,  55.64399590218565,  62.20067534638440,
     62.91145900291906,  53.13962804548330,  55.59366554715320,  62.61111814003368,
     62.66244814875670,  58.09339187838285,  61.11256111586631,  64.33235786384957,
     69.73045709795545,  69.10470715347692,  68.04107097516365,  66.14716986743504,
     67.09206937760580,  68.27619469022559,  72.18673121311446,  70.92374753095289,
     73.81255707253453,  74.15606719920338,  76.63182121761790,  78.46130604718124,
     65.56942938545296,  65.50906014325022,  70.81465644805017,  75.93250564723807,
     76.96752848836501,  76.19040080487545,  76.26711181813795,  77.80080601686956,
     78.29063787077104,  79.69081137979110,  79.05106205101127,  80.78800278268257,
     75.85016820748972,  78.28687266610947,  79.25500850260639,  77.01948560492802,
     68.99803231550435,  66.17293186820828,  65.60901519726411,  70.93244409151562,
     65.95051198398710,  69.84351415670412,  68.44292863529830,  61.48836294291115,
     52.69842167658372,  55.99237046258023,  60.90834400691658,  66.32101959631871,
     59.73995041261010,  61.43622755357976,  63.84441271603916,  64.56249988265114,
     68.19990155755167,  72.13190197794526,  72.55861260050784,  71.12982157760315,
     71.38342475267601,  70.01152859142937,  74.20846282498279,  74.43658255460164,
     78.67898911147611,  78.40565048322942,  79.59895733797759,  80.99925164167891,
     84.58372962773500,  77.70369287545617,  79.18367132610716,  75.95355181327594,
     79.08192541750374,  82.39181149601052,  82.71201146895177,  85.41699821999647,
     77.03716408491916,  74.57053790483886,  76.88980834577752,  78.75696030440729,
     74.73259515156042,  75.43965059165433,  76.40249131179854,  74.51090039476604,
     58.61185397894671,  65.92365954635456,  70.23272197874894,  69.56318967284210,
     52.95335220940760,  52.15944175967213,  59.91775125008380,  61.62766367540060,
     64.01822621937833,  58.60170472203898,  59.67974779585361,  57.61743646787124,
     63.00395490771065,  60.46348218221316,  60.52235378150319,  54.07443354160078,
     57.83111452814575,  54.38812233450692,  47.70847274198095,  37.52809978366949,
     32.46134205826088,  36.63036861347157,  28.65622410952068,  28.60014366248888,
     46.52591078662488,  42.26480998201676,  49.33178245742541,  44.99196967279641,
     43.36453249488635,  35.28677707076024,  44.60543992895871,  41.29493599113870,
     33.86567280977130,  45.42647163649778,  37.04167388355134,  40.56048115233136,
     39.18947948619932,  38.75869561420846,  35.32078743574289,  34.29416892058497,
     42.59896473250579,  42.25149629367897,  46.69940890160225,  43.69762621701994,
     46.16021118181348,  46.00084230167258,  41.98427277366395,  43.57827761681574,
     42.47773621604194,  50.52779852820042,  49.45195807591607,  51.86284512075687,
     52.56731640383381,  54.53056936947860,  59.29119216909293,  58.04856125579847,
     58.80446579522105,
};


TEST(momentum_backend, RSIDouble_SelfAllocated) {
    int window_size = 14;

    double* out = _RSI_DOUBLE(SAMPLE_CLOSE, NULL, data_len, window_size, 0);
    for (int i=0; i<data_len; i++) {
        ASSERT_DOUBLE_EQ(RSI_REF[i], out[i]);
    }

    free(out);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
