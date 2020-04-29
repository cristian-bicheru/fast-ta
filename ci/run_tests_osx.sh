simd=$(python3.7 -c "import detect_simd; print(detect_simd.detect())")
echo -e "\x1B[1m\e[34mDetected SIMD Capabilities:"
echo -e "\x1B[1m\e[34m"$simd

bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99"
sudo rm -r ~/.cache/bazel

if [[ $simd == *"'SSE2': 1"* ]]; then echo "SSE2 Support Detected, Running Tests..."; bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --define=SSE2=1 --copt="-msse2"; fi
sudo rm -r ~/.cache/bazel

if [[ $simd == *"'AVX': 1"* ]]; then echo "AVX Support Detected, Running Tests..."; bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --define=AVX=1 --copt="-mavx"; fi
sudo rm -r ~/.cache/bazel

if [[ $simd == *"'AVX512': 1"* ]]; then echo "AVX512 Support Detected, Running Tests..."; bazel test //fast_ta/src/... --noshow_loading_progress --noshow_progress --conlyopt="-std=c99" --define=AVX512=1 --copt="-mavx512f"; fi
