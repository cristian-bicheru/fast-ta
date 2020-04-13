cd fast_ta &&
rm -f *.so &&
python3 setup.py build_ext --inplace &&
cd .. &&
echo "--------------------------------------------" &&
python3 tests/tests.py
