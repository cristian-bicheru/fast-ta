cmake_minimum_required(VERSION 3.12)
project(fast-ta)

if (DEFINED SSE2)
    add_definitions(-DSSE2)
endif()

if (DEFINED SSE41)
    add_definitions(-DSSE2)
    add_definitions(-DSSE41)
endif()

if (DEFINED AVX)
    add_definitions(-DAVX)
endif()

if (DEFINED AVX2)
    add_definitions(-DAVX)
    add_definitions(-DAVX2)
endif()

if (DEFINED AVX512)
    add_definitions(-DAVX512)
endif()

find_package(Python3 COMPONENTS Interpreter Development)
if (NOT DEFINED Python3_INCLUDE_DIRS OR NOT DEFINED Python3_NumPy_INCLUDE_DIRS OR MSVC)
    execute_process(COMMAND "python" "scripts/find_includes.py"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE PYTHON_INCLUDES
            RESULT_VARIABLE FIND_RESULT)
    if (NOT FIND_RESULT EQUAL 0)
        execute_process(COMMAND "python3" "scripts/find_includes.py"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE PYTHON_INCLUDES
                RESULT_VARIABLE FIND_RESULT)
        if (NOT FIND_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to find python or numpy include directories!")
        endif()
    endif()
    if (MSVC)
        string(REPLACE "\\" "/" PYTHON_INCLUDES ${PYTHON_INCLUDES})
        string(REPLACE "\n" ";" PYTHON_INCLUDES ${PYTHON_INCLUDES})
        list(GET PYTHON_INCLUDES 2 PYTHON_LINK_DIR_OPTIMIZED)
        list(REMOVE_AT PYTHON_INCLUDES 2)
        list(GET PYTHON_INCLUDES 2 PYTHON_LINK_DIR_DEBUG)
        list(REMOVE_AT PYTHON_INCLUDES 2)
        set(PYTHON_LINK "optimized;${PYTHON_LINK_DIR_OPTIMIZED};debug;${PYTHON_LINK_DIR_DEBUG}")
    else()
        string(REPLACE "\n" ";" PYTHON_INCLUDES ${PYTHON_INCLUDES})
    endif()
else()
    set(PYTHON_INCLUDES "${Python3_INCLUDE_DIRS}; ${Python3_NumPy_INCLUDE_DIRS}")
endif()
list(GET PYTHON_INCLUDES 0 PY_INCLUDE)
message("Using Python headers at: ${PY_INCLUDE}")
list(GET PYTHON_INCLUDES 1 NUMPY_INCLUDE)
message("Using NumPy headers at: ${NUMPY_INCLUDE}")

set(CMAKE_C_STANDARD 11)
set(COMMON_BACKEND fast_ta/src/error_methods.c fast_ta/src/funcs/funcs.c fast_ta/src/funcs/funcs_unaligned.c fast_ta/src/2darray.c fast_ta/src/generic_simd/generic_simd.c)
if (DEFINED CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE EQUAL Debug)
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \Od ${Python3_LINK_OPTIONS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \Od ${Python3_LINK_OPTIONS}")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fno-inline -fno-inline-small-functions -fno-default-inline -march=native -mtune=native ${Python3_LINK_OPTIONS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fno-inline -fno-inline-small-functions -fno-default-inline -march=native -mtune=native ${Python3_LINK_OPTIONS}")
     endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /GS- /fp:fast /GL ${Python3_LINK_OPTIONS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /GS- /fp:fast /GL ${Python3_LINK_OPTIONS}")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mtune=native -fomit-frame-pointer ${Python3_LINK_OPTIONS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -fomit-frame-pointer ${Python3_LINK_OPTIONS}")
    endif()
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
message("Using link options: ${Python3_LINK_OPTIONS}")

add_library(core SHARED ${COMMON_BACKEND} fast_ta/src/core.c fast_ta/src/core/core_backend.c)
add_library(momentum SHARED ${COMMON_BACKEND} fast_ta/src/momentum.c fast_ta/src/momentum/momentum_backend.c)
add_library(volume SHARED ${COMMON_BACKEND} fast_ta/src/volume.c fast_ta/src/volume/volume_backend.c)
add_library(volatility SHARED ${COMMON_BACKEND} fast_ta/src/volatility.c fast_ta/src/volatility/volatility_backend.c)
target_include_directories(core PRIVATE ${PYTHON_INCLUDES})
target_include_directories(momentum PRIVATE ${PYTHON_INCLUDES})
target_include_directories(volume PRIVATE ${PYTHON_INCLUDES})
target_include_directories(volatility PRIVATE ${PYTHON_INCLUDES})

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

if (DEFINED CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE EQUAL Debug)
    include(CodeCoverage.cmake)
    APPEND_COVERAGE_COMPILER_FLAGS()
endif()

add_executable(momentum_test fast_ta/src/momentum_backend.test.cpp ${COMMON_BACKEND} fast_ta/src/momentum/momentum_backend.c)
target_include_directories(momentum_test PRIVATE ${PYTHON_INCLUDES})
target_link_libraries(momentum_test gtest_main)
add_test(NAME momentum_test COMMAND momentum_test)

add_executable(volatility_test fast_ta/src/volatility_backend.test.cpp ${COMMON_BACKEND} fast_ta/src/volatility/volatility_backend.c)
target_include_directories(volatility_test PRIVATE ${PYTHON_INCLUDES})
target_link_libraries(volatility_test gtest_main)
add_test(NAME volatility_test COMMAND volatility_test)

add_executable(volume_test fast_ta/src/volume_backend.test.cpp ${COMMON_BACKEND} fast_ta/src/volume/volume_backend.c)
target_include_directories(volume_test PRIVATE ${PYTHON_INCLUDES})
target_link_libraries(volume_test gtest_main)
add_test(NAME volume_test COMMAND volume_test)

if (MSVC)
    message("Linking Python Debug Binaries...")
    target_link_libraries(core ${PYTHON_LINK})
    target_link_libraries(momentum ${PYTHON_LINK})
    target_link_libraries(volume ${PYTHON_LINK})
    target_link_libraries(volatility ${PYTHON_LINK})
    target_link_libraries(momentum_test ${PYTHON_LINK})
    target_link_libraries(volatility_test ${PYTHON_LINK})
    target_link_libraries(volume_test ${PYTHON_LINK})
    message("Linked against ${PYTHON_LINK}")
 endif()