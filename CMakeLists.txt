cmake_minimum_required(VERSION 3.1)
project(fast-ta)

if (DEFINED SSE2)
    add_definitions(-DSSE2)
endif()

if (DEFINED AVX)
    add_definitions(-DAVX)
endif()

if (DEFINED AVX512)
    add_definitions(-DAVX512)
endif()

find_package(Python3 COMPONENTS Interpreter Development)

set(CMAKE_C_STANDARD 11)
set(COMMON_BACKEND fast_ta/src/error_methods.c fast_ta/src/funcs/funcs.c fast_ta/src/funcs/funcs_unaligned.c fast_ta/src/2darray.c fast_ta/src/generic_simd/generic_simd.c)
if (DEFINED CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE EQUAL Debug)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fno-inline -fno-inline-small-functions -fno-default-inline -march=native -mtune=native ${Python3_LINK_OPTIONS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fno-inline -fno-inline-small-functions -fno-default-inline -march=native -mtune=native ${Python3_LINK_OPTIONS}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mtune=native -fomit-frame-pointer ${Python3_LINK_OPTIONS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -fomit-frame-pointer ${Python3_LINK_OPTIONS}")
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

add_library(core SHARED ${COMMON_BACKEND} fast_ta/src/core.c fast_ta/src/core/core_backend.c)
add_library(momentum SHARED ${COMMON_BACKEND} fast_ta/src/momentum.c fast_ta/src/momentum/momentum_backend.c)
add_library(volume SHARED ${COMMON_BACKEND} fast_ta/src/volume.c fast_ta/src/volume/volume_backend.c)
add_library(volatility SHARED ${COMMON_BACKEND} fast_ta/src/volatility.c fast_ta/src/volatility/volatility_backend.c)
target_include_directories(core PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(momentum PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(volume PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(volatility PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

include(CodeCoverage.cmake)
APPEND_COVERAGE_COMPILER_FLAGS()

add_executable(momentum_test fast_ta/src/momentum_backend.test.cpp ${COMMON_BACKEND} fast_ta/src/momentum/momentum_backend.c)
target_include_directories(momentum_test PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_libraries(momentum_test gtest_main)
add_test(NAME momentum_test COMMAND momentum_test)

add_executable(volatility_test fast_ta/src/volatility_backend.test.cpp ${COMMON_BACKEND} fast_ta/src/volatility/volatility_backend.c)
target_include_directories(volatility_test PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_libraries(volatility_test gtest_main)
add_test(NAME volatility_test COMMAND volatility_test)

add_executable(volume_test fast_ta/src/volume_backend.test.cpp ${COMMON_BACKEND} fast_ta/src/volume/volume_backend.c)
target_include_directories(volume_test PRIVATE ${Python3_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_libraries(volume_test gtest_main)
add_test(NAME volume_test COMMAND volume_test)