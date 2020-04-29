simd=$(python3.7 -c "import detect_simd; print(detect_simd.detect())")
echo -e "\e[1m\e[34mDetected SIMD Capabilities:"
echo -e "\e[1m\e[34m"$simd

bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --collect_code_coverage
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/momentum_backend_tests/coverage.dat no_simd1.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volume_backend_tests/coverage.dat no_simd2.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volatility_backend_tests/coverage.dat no_simd3.dat
sudo rm -r ~/.cache/bazel

if [[ $simd == *"'SSE2': 1"* ]]; then echo "SSE2 Support Detected, Running Tests..."; bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --define=SSE2=1 --copt="-msse2" --collect_code_coverage; fi
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/momentum_backend_tests/coverage.dat sse21.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volume_backend_tests/coverage.dat sse22.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volatility_backend_tests/coverage.dat sse23.dat
sudo rm -r ~/.cache/bazel

if [[ $simd == *"'AVX': 1"* ]]; then echo "AVX Support Detected, Running Tests..."; bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --define=AVX=1 --copt="-mavx" --collect_code_coverage; fi
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/momentum_backend_tests/coverage.dat avx1.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volume_backend_tests/coverage.dat avx2.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volatility_backend_tests/coverage.dat avx3.dat
sudo rm -r ~/.cache/bazel

if [[ $simd == *"'AVX512': 1"* ]]; then echo "AVX512 Support Detected, Running Tests..."; bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --define=AVX512=1 --copt="-mavx512f" --collect_code_coverage; fi
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/momentum_backend_tests/coverage.dat avx5121.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volume_backend_tests/coverage.dat avx5122.dat
mv bazel-out/k8-fastbuild/testlogs/fast_ta/src/volatility_backend_tests/coverage.dat avx5123.dat
