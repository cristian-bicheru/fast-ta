cd ../

curl --connect-timeout 10 --max-time 10 --retry 5 --retry-delay 0 --retry-max-time 120 https://codecov.io/bash -o uploader.sh
chmod +x uploader.sh
./uploader.sh -f no_simd1.dat
./uploader.sh -f no_simd2.dat
./uploader.sh -f no_simd3.dat
if [[ $simd == *"'SSE2': 1"* ]]; then ./uploader.sh -f sse21.dat; ./uploader.sh -f sse22.dat; ./uploader.sh -f sse23.dat; fi
if [[ $simd == *"'AVX': 1"* ]]; then ./uploader.sh -f avx1.dat; ./uploader.sh -f avx2.dat; ./uploader.sh -f avx3.dat; fi
if [[ $simd == *"'AVX512': 1"* ]]; then ./uploader.sh -f avx5121.dat; ./uploader.sh -f avx5122.dat; ./uploader.sh -f avx5123.dat; fi
