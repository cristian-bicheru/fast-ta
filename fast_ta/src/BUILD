package(default_visibility = ["//fast_ta:__subpackages__",
                              "//tests:__subpackages__"])

# this will vary from system to system. The best/simplist solution currently
# is for each fast_ta developer to switch this to their own path.
PYTHON_INCLUDE = [
    "-isystem external/python",
    "-isystem external/include",
    "-isystem external/localinclude",
]

PYTHON_DEPS = [
    "@python//:headers",
    "@include//:headers",
    "@localinclude//:headers",
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
    deps = PYTHON_DEPS,
)

cc_library(
    name = "error_methods",
    srcs = ["error_methods.c"],
    hdrs = ["error_methods.h"],
    copts = PYTHON_INCLUDE,
    deps = PYTHON_DEPS,
)

cc_library(
    name = "generic_simd",
    srcs = ["generic_simd/generic_simd.c"],
    hdrs = ["generic_simd/generic_simd.h"],
)

cc_library(
    name = "testing_common",
    hdrs = ["testing_common.h"],
    data = [
	"//fast_ta/src/test_data:sample_close_data.txt",
	"//fast_ta/src/test_data:sample_high_data.txt",
	"//fast_ta/src/test_data:sample_low_data.txt",
	"//fast_ta/src/test_data:sample_open_data.txt",
	"//fast_ta/src/test_data:sample_volume_data.txt",
	"//fast_ta/src/test_data:rsi_ref.txt",
	"//fast_ta/src/test_data:ao_ref.txt",
	"//fast_ta/src/test_data:kama_ref.txt",
	"//fast_ta/src/test_data:roc_ref.txt",
	"//fast_ta/src/test_data:stoch_ref.txt",
	"//fast_ta/src/test_data:stoch_signal_ref.txt",
	"//fast_ta/src/test_data:tsi_ref.txt",
	"//fast_ta/src/test_data:uo_ref.txt",
	"//fast_ta/src/test_data:wr_ref.txt",
	"//fast_ta/src/test_data:atr_ref.txt",
	"//fast_ta/src/test_data:boll_ref.txt",
	"//fast_ta/src/test_data:bolm_ref.txt",
	"//fast_ta/src/test_data:bolh_ref.txt",
	"//fast_ta/src/test_data:adi_ref.txt",
	"//fast_ta/src/test_data:cmf_ref.txt",
	"//fast_ta/src/test_data:emv_ref.txt",
	"//fast_ta/src/test_data:emv_sma_ref.txt",
	"//fast_ta/src/test_data:fi_ref.txt",
	"//fast_ta/src/test_data:mfi_ref.txt",
	"//fast_ta/src/test_data:nvi_ref.txt",
	"//fast_ta/src/test_data:obv_ref.txt",
	"//fast_ta/src/test_data:vpt_ref.txt",
	"//fast_ta/src/test_data:vwap_ref.txt",
	"//fast_ta/src/test_data:dcl_ref.txt",
	"//fast_ta/src/test_data:dcm_ref.txt",
	"//fast_ta/src/test_data:dch_ref.txt",
	"//fast_ta/src/test_data:kcl_ref.txt",
	"//fast_ta/src/test_data:kcm_ref.txt",
	"//fast_ta/src/test_data:kch_ref.txt",
])

##################### Core Library #####################
cc_library(
    name = "funcs",
    srcs = ["funcs/funcs.c"],
    hdrs = ["funcs/funcs.h"],
    deps = [
        ":generic_simd",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE
)

cc_library(
    name = "core_backend",
    srcs = ["core/core_backend.c"],
    hdrs = ["core/core_backend.h"],
    deps = [
        ":generic_simd",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE
)

cc_library(
    name = "momentum_backend",
    srcs = ["momentum/momentum_backend.c"],
    hdrs = ["momentum/momentum_backend.h"],
    deps = [
        ":funcs",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volume_backend",
    srcs = ["volume/volume_backend.c"],
    hdrs = ["volume/volume_backend.h"],
    deps = [
        ":funcs",
        ":2darray",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volatility_backend",
    srcs = ["volatility/volatility_backend.c"],
    hdrs = ["volatility/volatility_backend.h"],
    deps = [
        ":funcs",
        ":2darray",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)

cc_test(
    name = "momentum_backend_tests",
    srcs = ["momentum_backend.test.cpp"],
    copts = ["-isystem external/gtest/include"],
    deps = [
        "@gtest//:main",
        ":momentum_backend",
        ":testing_common",
    ],
)

cc_test(
    name = "volume_backend_tests",
    srcs = ["volume_backend.test.cpp"],
    copts = ["-isystem external/gtest/include"],
    deps = [
        "@gtest//:main",
        ":volume_backend",
        ":testing_common",
    ],
)

cc_test(
    name = "volatility_backend_tests",
    srcs = ["volatility_backend.test.cpp"],
    copts = ["-isystem external/gtest/include"],
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
    name = "core",
    srcs = ["core.c"],
    deps = [
        ":core_backend",
        ":error_methods",
        ":2darray",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "momentum",
    srcs = ["momentum.c"],
    deps = [
        ":momentum_backend",
        ":error_methods",
        ":2darray",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volume",
    srcs = ["volume.c"],
    deps = [
        ":volume_backend",
        ":error_methods",
        ":2darray",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)

cc_library(
    name = "volatility",
    srcs = ["volatility.c"],
    deps = [
        ":volatility_backend",
        ":error_methods",
	":2darray",
    ]+PYTHON_DEPS,
    copts = PYTHON_INCLUDE,
)
