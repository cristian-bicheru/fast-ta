wget https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py
python3.7 -m pip install -U pip
python3.7 -m pip install --upgrade setuptools
python3.7 -m pip install --upgrade cython
python3.7 -m pip install numpy detect-simd
dest=$(python3.7 -c "import numpy; print(numpy.get_include()+'/numpy')")
sudo ln -sfn $dest /usr/local/include/numpy
pyincludes=$(python3.7-config --includes)
pyinc=$(echo $pyincludes | tr " " "\n")
pyinc=(${pyinc[0]})
pyinc=(${pyinc:2})
