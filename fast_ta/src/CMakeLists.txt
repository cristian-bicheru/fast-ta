cmake_minimum_required(VERSION 3.15)
project(fast_ta C)
include_directories(/usr/include/python3.7/)

set(CMAKE_C_STANDARD 99)

add_library(fast_ta SHARED momentum.c momentum_backend.c funcs.c debug_tools.c error_methods.c)