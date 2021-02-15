simd=$(python3.7 -c "import detect_simd; print(detect_simd.detect())")
echo -e "\e[1m\e[34mDetected SIMD Capabilities:"
echo -e "\e[1m\e[34m"$simd

echo -e "\e[1m\e[34mTesting with Clang..."
export CC=clang

mkdir build && cd build
cmake ..
make -j
cd ..
./build/momentum_test
./build/volatility_test
./build/volume_test
rm -r build/*

if [[ $simd == *"'SSE2': 1"* ]]
then
  echo "SSE2 Support Detected, Running Tests...";
  cd build
  cmake -DSSE2=1 ..
  make -j
  cd ..
  ./build/momentum_test
  ./build/volatility_test
  ./build/volume_test
fi
rm -r build/*

if [[ $simd == *"'AVX': 1"* ]]
then
  echo "AVX Support Detected, Running Tests...";
  cd build
  cmake -DAVX=1 ..
  make -j
  cd ..
  ./build/momentum_test
  ./build/volatility_test
  ./build/volume_test
fi
rm -r build/*

if [[ $simd == *"'AVX512': 1"* ]]
then
  echo "AVX512 Support Detected, Running Tests...";
  cd build
  cmake -DAVX512=1 ..
  make -j
  cd ..
  ./build/momentum_test
  ./build/volatility_test
  ./build/volume_test
fi
rm -r build/*
