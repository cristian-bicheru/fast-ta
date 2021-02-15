wget https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py
python3.7 -m pip install -U pip
python3.7 -m pip install --upgrade setuptools
python3.7 -m pip install --upgrade cython
python3.7 -m pip install numpy detect-simd
dest=$(python3.7 -c "import numpy; print(numpy.get_include()+'/numpy')")
sudo ln -sfn $dest /usr/include/numpy
