enable_testing()

set(TARGET_NAME ${PROJECT_NAME}_UnitTest)
message(STATUS ${SEP} ${TARGET_NAME} ${SEP})

find_package(Armadillo REQUIRED)
cmake_policy(SET CMP0167 NEW)
find_package(Boost 1.89 REQUIRED)
find_package(Boost COMPONENTS program_options log log_setup REQUIRED)

set(ARMADILLO_CONFIG
    ARMA_NO_DEBUG
    ARMA_BLAS_LONG
    ARMA_DONT_USE_STD_MUTEX
    ARMA_DONT_PRINT_EXCEPTIONS
    ARMA_MAT_PREALLOC=32
    ARMA_DONT_ZERO_INIT
    ARMA_USE_ARPACK=OFF
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64
)

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
)

source_group("Source Files\\Source" FILES ${SourceFiles})

add_executable(${TARGET_NAME} ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${TARGET_NAME}
    PRIVATE ${ARMADILLO_CONFIG}
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
    SYSTEM PRIVATE ${ARMADILLO_INCLUDE_DIRS}
    SYSTEM PRIVATE ${FFT_INCLUDE_DIR}
    PRIVATE ../Source
    PRIVATE ../Include
)

target_link_libraries(${TARGET_NAME}
    PRIVATE "-framework Accelerate"
    PRIVATE ${FFT_LINK_CMD}
    PRIVATE Eigen3::Eigen
    PRIVATE ${PROJECT_NAME}
)

set_target_properties(${TARGET_NAME} 
    PROPERTIES
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_ARGUMENTS --color_output=no
)