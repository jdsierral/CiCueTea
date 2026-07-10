cmake_minimum_required(VERSION 3.30)


set(SEP " ========= ")
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0")
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION CURRENT_VERSION)

project("CiCueTea" VERSION ${CURRENT_VERSION})
set(TARGET_NAME ${PROJECT_NAME})
message(STATUS ${SEP} ${TARGET_NAME} ${SEP})
option(BUILD_TESTS "Build all the unit tests" OFF)

# Eigen's package config only matches same-major requests, so ask for any
# version and enforce the floor ourselves (3.4+ and 5.x both work).
find_package(Eigen3 REQUIRED)
if(Eigen3_VERSION VERSION_LESS 3.4)
    message(FATAL_ERROR "CiCueTea requires Eigen >= 3.4 (found ${Eigen3_VERSION})")
endif()
message(STATUS "Eigen version: ${Eigen3_VERSION}")

include(SourceFiles.cmake)
include(FFTSelection.cmake)

set(EIGEN_CONFIG
    # EIGEN_USE_BLAS
    # EIGEN_USE_LAPACKE
    PRIVATE EIGEN_RUNTIME_NO_MALLOC
)

add_library(${TARGET_NAME} STATIC ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /W4)
else()
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(${TARGET_NAME}
    PRIVATE ${EIGEN_CONFIG}
    PRIVATE ${FFT_NAME}
)

target_include_directories(${TARGET_NAME}
    PRIVATE Include/
    SYSTEM PRIVATE ${FFT_INCLUDE_DIR}
)

target_link_libraries(${TARGET_NAME} 
    PRIVATE ${FFT_LINK_CMD}
    PRIVATE Eigen3::Eigen
)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
