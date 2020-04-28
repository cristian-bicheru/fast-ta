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
    name = "momentum_backend_args",
    hdrs = ["momentum_backend_args.h"],
)

##################### Utils #####################
cc_library(
    name = "2darray",
    srcs = ["2darray.c"],
    hdrs = ["2darray.h"],
)

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
    srcs = ["generic_simd.c"],
    hdrs = ["generic_simd.h"],
)

cc_library(
    name = "testing_common",
    hdrs = ["testing_common.h"],
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
    ],
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volume_backend",
    srcs = ["volume_backend.c"],
    hdrs = ["volume_backend.h"],
    deps = [
        ":funcs",
        ":2darray",
    ],
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volatility_backend",
    srcs = ["volatility_backend.c"],
    hdrs = ["volatility_backend.h"],
    deps = [
        ":funcs",
        ":2darray",
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
        ":testing_common",
    ],
)

cc_test(
    name = "volume_backend_tests",
    srcs = ["volume_backend.test.cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@gtest//:main",
        ":volume_backend",
        ":testing_common",
    ],
)

cc_test(
    name = "volatility_backend_tests",
    srcs = ["volatility_backend.test.cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@gtest//:main",
        ":volatility_backend",
        ":testing_common",
	":2darray",
    ],
)

cc_library(
    name = "parallel_momentum_backend",
    srcs = ["parallel_momentum_backend.c"],
    hdrs = ["parallel_momentum_backend.h"],
    deps = [
        ":momentum_backend",
        ":2darray",
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

cc_library(
    name = "volume",
    srcs = ["volume.c"],
    deps = [
        ":volume_backend",
        ":error_methods",
        ":2darray",
    ],
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volatility",
    srcs = ["volatility.c"],
    deps = [
        ":volatility_backend",
        ":error_methods",
	":2darray",
    ],
    copts = PYTHON_INCLUDE,
)
