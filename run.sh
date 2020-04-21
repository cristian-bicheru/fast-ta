rm -f *.so &&
python3 setup.py build_ext --inplace &&
echo "--------------------------------------------" &&
python3 tests/tests.py --save-plots
