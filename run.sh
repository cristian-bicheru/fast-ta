cd fast_ta &&
rm -f *.so &&
python3.7 setup.py build_ext --inplace &&
cd .. &&
echo "--------------------------------------------" &&
python3.7 tests/tests.py
