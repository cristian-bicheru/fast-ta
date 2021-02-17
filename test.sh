mkdir test_build && cd test_build
cmake ..
make -j
cd ..
./test_build/momentum_test
./test_build/volume_test
./test_build/volatility_test
rm -rf test_build
