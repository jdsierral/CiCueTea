# Standalone C++ benchmark: CiCueTea vs (optionally) The Gaborator.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -DGABORATOR_DIR=/path/to/gaborator-2.1        # optional
#   cmake --build build -j && ./build/cqt_bench
#
# The Gaborator is AGPLv3-or-commercial and is therefore not vendored, fetched,
# or linked by default: download it from https://www.gaborator.com and point
# GABORATOR_DIR at the unpacked source to enable the comparison. Everything it
# needs is header-only. On macOS it is given vDSP (its fastest backend, and the
# same FFT CiCueTea uses by default) so the comparison is fair.

cmake_minimum_required(VERSION 3.22)
project(CiCueTeaBench C CXX) # C for rt-cqt's pffft

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cicuetea)
find_package(Eigen3 REQUIRED)

add_executable(cqt_bench bench.cpp)
target_compile_features(cqt_bench PRIVATE cxx_std_20)
target_include_directories(cqt_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../Include)
target_link_libraries(cqt_bench PRIVATE CiCueTea Eigen3::Eigen)

# Configure-time facts baked in as macros: this tool is compiled and run on
# the same machine, so CMake is the authority on CPU, compiler, and where the
# results folder lives (the binary can't know the source tree otherwise).
cmake_host_system_information(RESULT BENCH_CPU QUERY PROCESSOR_NAME)
if(NOT BENCH_CPU OR BENCH_CPU STREQUAL "unknown")
    cmake_host_system_information(RESULT BENCH_CPU QUERY PROCESSOR_DESCRIPTION)
endif()
target_compile_definitions(cqt_bench PRIVATE
    BENCH_CPU_NAME="${BENCH_CPU}"
    BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    BENCH_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../results"
)

if(GABORATOR_DIR)
    target_include_directories(cqt_bench SYSTEM PRIVATE ${GABORATOR_DIR})
    target_compile_definitions(cqt_bench PRIVATE HAVE_GABORATOR)
    if(APPLE)
        target_compile_definitions(cqt_bench PRIVATE GABORATOR_USE_VDSP=1)
        target_link_libraries(cqt_bench PRIVATE "-framework Accelerate")
    endif()
    message(STATUS "Gaborator comparison enabled: ${GABORATOR_DIR}")
else()
    message(STATUS "Gaborator comparison disabled (set -DGABORATOR_DIR to enable)")
endif()

# rt-cqt (BSD-3) — point RTCQT_DIR at a recursive clone of
# https://github.com/jmerkt/rt-cqt (needs its pffft submodule).
if(RTCQT_DIR)
    target_include_directories(cqt_bench SYSTEM PRIVATE ${RTCQT_DIR}/include)
    target_sources(cqt_bench PRIVATE
        ${RTCQT_DIR}/submodules/pffft/pffft.c
        ${RTCQT_DIR}/submodules/pffft/pffft_common.c
        ${RTCQT_DIR}/submodules/pffft/pffft_double.c
    )
    target_compile_definitions(cqt_bench PRIVATE HAVE_RTCQT)
    message(STATUS "rt-cqt comparison enabled: ${RTCQT_DIR}")
else()
    message(STATUS "rt-cqt comparison disabled (set -DRTCQT_DIR to enable)")
endif()
