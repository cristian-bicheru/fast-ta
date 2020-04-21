[![Build Status](https://travis-ci.com/cristian-bicheru/fast-ta.svg?branch=master)](https://travis-ci.com/cristian-bicheru/fast-ta)

# Building:
```bash
python3.x setup.py build_ext --inplace
```

# Testing:

To run CI tests:
```bash
bazel test //fast_ta/src/...
```
All of these must pass for any code to be added to the repo. Info on installing
bazel [here](https://docs.bazel.build/versions/master/install.html). More
info on developing & testing can be found in
[CONTRIBUTING.md](https://github.com/cristian-bicheru/fast-ta/blob/master/CONTRIBUTING.md).

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
