curl -LO https://github.com/bazelbuild/bazel/releases/download/3.0.0/bazel-3.0.0-installer-darwin-x86_64.sh
chmod +x bazel-3.0.0-installer-darwin-x86_64.sh
./bazel-3.0.0-installer-darwin-x86_64.sh --user
export PATH="$PATH:$HOME/bin"

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

cat > WORKSPACE <<- EOM
workspace(name = 'fast_ta')

new_local_repository(
    name = "python",
    path = "$pyinc",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"])
)
"""
)

new_local_repository(
    name = "include",
    path = "/usr/include",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"])
)
"""
)

new_local_repository(
    name = "localinclude",
    path = "/usr/local/include",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"])
)
"""
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)
EOM
