set(TARGET_NAME ${PROJECT_NAME}_UnitTest)
message(STATUS ${SEP} ${TARGET_NAME} ${SEP})

cmake_policy(SET CMP0167 NEW)
find_package(Boost 1.89 REQUIRED)
find_package(Boost COMPONENTS program_options log log_setup REQUIRED)

set(SourceFiles
    Source/main.cpp
    Source/DFT_UnitTests.cpp
    Source/StabilityTest.cpp
    Source/CQT_UnitTests.cpp
    Source/Slicing_UnitTests.cpp
    Source/SlidingCQT_UnitTests.cpp
    Source/Perf_UnitTests.cpp
    Source/SlidingExtended_UnitTests.cpp
    Source/SliCQTTest_UnitTests.cpp
    Source/FFTLib_UnitTests.cpp
    Source/Timing_Tests.cpp
    Source/VectorOps.h
    Source/EmptyCQTProc.h
    Source/Benchtools.h
)

source_group("Source Files\\Source" FILES ${SourceFiles})

add_executable(${TARGET_NAME} ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${TARGET_NAME}
    PRIVATE ${EIGEN_CONFIG}
)

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -fcolor-diagnostics
)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE ${Boost_INCLUDE_DIRS}
    SYSTEM PRIVATE ${FFT_INCLUDE_DIR}
    PRIVATE ../Source
    PRIVATE ../Include
)

target_link_libraries(${TARGET_NAME}
    PRIVATE ${FFT_LINK_CMD}
    PRIVATE Eigen3::Eigen
    PRIVATE ${PROJECT_NAME}
)

set_target_properties(${TARGET_NAME}
    PROPERTIES
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_ARGUMENTS --color_output=no
)

# CTest registration. Boost.Test has no shipped auto-discovery, so test cases
# are grouped by their per-file name prefixes and run via --run_test filters.
# A new BOOST_AUTO_TEST_CASE is picked up automatically only if its name
# matches an existing filter below — otherwise add it to one.
function(add_boost_test GROUP FILTER)
    add_test(NAME ${GROUP} COMMAND ${TARGET_NAME} --run_test=${FILTER} --color_output=no)
    set_tests_properties(${GROUP} PROPERTIES TIMEOUT 120)
endfunction()

add_boost_test(DFT              "DFTTest*")
add_boost_test(CQT              "CQTTest*,SlidingCQT")
add_boost_test(Slicing          "Slicing*,CQTSlicing*")
add_boost_test(OlaProcessors    "OlaProc*")
add_boost_test(SlidingExtended  "DoubleBufferTest,ExtendedTest*")
add_boost_test(SliCQT           "SparseCQT*,SliCQTTest1")
add_boost_test(Stability        "BenchTest")

# Benchmarks: slow, timing-dependent — excluded from quick runs via `ctest -LE bench`
add_boost_test(FFTBench         "FFTLibTest*")
add_boost_test(Perf             "perf*")
add_boost_test(Timing           "BenchmarkTest1")

set_tests_properties(FFTBench Perf Timing PROPERTIES LABELS "bench" TIMEOUT 600)