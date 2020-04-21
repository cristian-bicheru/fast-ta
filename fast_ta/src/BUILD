package(default_visibility = ["//fast_ta:__subpackages__",
                              "//tests:__subpackages__"])

# this will vary from system to system. The best/simplist solution currently
# is for each fast_ta developer to switch this to their own path.
PYTHON_INCLUDE = [
    "-I/usr/include/python3.7",
    "-I/usr/include/",
]

##################### Structs #####################
cc_library(
    name = "array_pair",
    hdrs = ["array_pair.h"],
)

cc_library(
    name = "momentum_backend_args",
    hdrs = ["momentum_backend_args.h"],
)

##################### Utils #####################
cc_library(
    name = "debug_tools",
    srcs = ["debug_tools.c"],
    hdrs = ["debug_tools.h"],
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "error_methods",
    srcs = ["error_methods.c"],
    hdrs = ["error_methods.h"],
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "generic_simd",
    hdrs = ["generic_simd.h"],
)

##################### Core Library #####################
cc_library(
    name = "funcs",
    srcs = ["funcs.c"],
    hdrs = ["funcs.h"],
    deps = [
        ":generic_simd",
    ],
    copts = PYTHON_INCLUDE
)

cc_library(
    name = "momentum_backend",
    srcs = ["momentum_backend.c"],
    hdrs = ["momentum_backend.h"],
    deps = [
        ":funcs",
        ":array_pair",
    ],
    copts = PYTHON_INCLUDE,
)

cc_test(
    name = "momentum_backend_tests",
    srcs = ["momentum_backend.test.cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@gtest//:main",
        ":momentum_backend",
    ],
)

cc_library(
    name = "parallel_momentum_backend",
    srcs = ["parallel_momentum_backend.c"],
    hdrs = ["parallel_momentum_backend.h"],
    deps = [
        ":momentum_backend",
        ":array_pair",
        ":momentum_backend_args",
        ":error_methods",
    ],
    linkopts = [
        "-lpthread",
    ],
)

cc_library(
    name = "momentum",
    srcs = ["momentum.c"],
    deps = [
        ":momentum_backend",
        ":parallel_momentum_backend",
    ],
    copts = PYTHON_INCLUDE,
)
