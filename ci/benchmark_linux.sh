simd=$(python3.7 -c "import detect_simd; print(detect_simd.detect())")
rm fast_ta/*.so

echo -e "\e[1m\e[34mBenchmarking without SIMD..."
python3.7 setup_custom_arch.py build_ext --inplace > /dev/null
cd benchmarks
python3.7 throughput_fast_ta.py > /dev/null
cd results
csvtool readable throughput_fast_ta.csv
cd ../../
rm fast_ta/*.so


if [[ $simd == *"'SSE2': 1"* ]]
then
echo -e "\e[1m\e[34mBenchmarking with SSE2..."
python3.7 setup_custom_arch.py build_ext --inplace -DSSE2 > /dev/null
cd benchmarks
python3.7 throughput_fast_ta.py > /dev/null
cd results
csvtool readable throughput_fast_ta.csv
cd ../../
rm fast_ta/*.so
fi


if [[ $simd == *"'AVX': 1"* ]]
then
echo -e "\e[1m\e[34mBenchmarking with AVX..."
python3.7 setup_custom_arch.py build_ext --inplace -DAVX > /dev/null
cd benchmarks
python3.7 throughput_fast_ta.py > /dev/null
cd results
csvtool readable throughput_fast_ta.csv
cd ../../
rm fast_ta/*.so
fi

if [[ $simd == *"'AVX512': 1"* ]]
then
echo -e "\e[1m\e[34mBenchmarking with AVX512..."
python3.7 setup_custom_arch.py build_ext --inplace -DAVX512 > /dev/null
cd benchmarks
python3.7 throughput_fast_ta.py > /dev/null
cd results
csvtool readable throughput_fast_ta.csv
cd ../../
fi
