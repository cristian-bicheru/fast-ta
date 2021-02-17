[![Build Status](https://travis-ci.com/cristian-bicheru/fast-ta.svg?branch=master)](https://travis-ci.com/cristian-bicheru/fast-ta)
[![codecov](https://codecov.io/gh/cristian-bicheru/fast-ta/branch/master/graph/badge.svg)](https://codecov.io/gh/cristian-bicheru/fast-ta)

# Cloning Source Code:
```bash
git clone https://github.com/cristian-bicheru/fast-ta.git && cd fast-ta && git submodule update --init
```

# Building:
```bash
python3.x setup.py build_ext --inplace
```
OR using CMake
```bash
mkdir test_build && cd test_build
cmake -D<arch>=1 ..
make -j
``
For debugging purposes, you can use `cmake -DCMAKE_BUILD_TYPE=Debug -D<arch>=1 ..` (where `<arch>` is either SSE2, SSE41, AVX, AVX2, or AVX512, or you can omit the `-D<arch>=1` entirely for a SIMD-free build.) 
NOTE: without `-DCMAKE_BUILD_TYPE=Debug` the compiler may introduce SIMD optimizations.

Building with MSVC:
```bash
mkdir test_build && cd test_build
cmake ..
msbuild fast-ta.sln
```
NOTE: This requires msbuild to be in PATH, also make sure the selected Python distribution was installed with debug binaries. If not, re-run the installer and tick the option.

# Testing:

To run CI tests:
```bash
./test.sh
```
All of these must pass for any code to be added to the repo.

For general, eyeball testing you can generate plots of the indicators with
this script.
```bash
python3.x tests/tests.py --show-plots --save-plots
```

# Benchmarks:
**REQUIRES:** ta
```bash
python3.x benchmarks/<indicator>.py
```
**OUTPUTS:** SVG plotting times speedup

# Useful Resources:
https://docs.python.org/3/c-api/index.html

https://docs.scipy.org/doc/numpy/reference/c-api.html

https://db.in.tum.de/~finis/x86-intrin-cheatsheet-v2.1.pdf

# TODO:

 - Aggregate code coverage data in CI.
