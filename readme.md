# Building:
```bash
python3.x setup.py build_ext --inplace
```

# Testing:
```bash
cd tests && python3.x tests.py
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
