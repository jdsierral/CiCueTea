cmake_minimum_required(VERSION 4.0)

set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Support macOS down to Big Sur")
include(GitData.cmake)
project("CiCueTea" VERSION ${CURRENT_VERSION})
set(TARGET_NAME ${PROJECT_NAME})
message(STATUS ${SEP} ${TARGET_NAME} ${SEP})
option(BUILD_TESTS "Build all the unit tests" OFF)

find_package(Eigen3 3.3 REQUIRED)

include(SourceFiles.cmake)
include(FFTSelection.cmake)

set(EIGEN_CONFIG
    # EIGEN_USE_BLAS
    # EIGEN_USE_LAPACKE
    PRIVATE EIGEN_RUNTIME_NO_MALLOC
)

add_library(${TARGET_NAME} STATIC ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_definitions(${TARGET_NAME}
    PRIVATE ${EIGEN_CONFIG}
    PRIVATE ${FFT_NAME}
    PRIVATE CURRENT_GIT_COMMIT=${CURRENT_GIT_COMMIT}
    PRIVATE CURRENT_GIT_TAG=${CURRENT_GIT_TAG}
    PRIVATE CURRENT_GIT_BRANCH=${CURRENT_GIT_BRANCH}
    PRIVATE CURRENT_GIT_COMMIT_MESSAGE=${CURRENT_GIT_COMMIT_MESSAGE}
    PRIVATE CURRENT_GIT_COMMIT_DATE=${CURRENT_GIT_COMMIT_DATE}
    PRIVATE CURRENT_GIT_COMMIT_TIME=${CURRENT_GIT_COMMIT_TIME}
    PRIVATE CURRENT_VERSION=${CURRENT_VERSION}
)

target_include_directories(${TARGET_NAME}
    PRIVATE Include/
    SYSTEM PRIVATE ${FFT_INCLUDE_DIR}
)

set_target_properties(${TARGET_NAME} 
    PROPERTIES 
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_LAUNCH_CONFIGURATION "Release"
        XCODE_SCHEME_ARGUMENTS --color_output=no
        XCODE_SCHEME_ENVIRONMENT "PATH=/opt/homebrew/bin"
)

target_link_libraries(${TARGET_NAME} 
    PRIVATE ${FFT_LINK_CMD}
    PRIVATE Eigen3::Eigen
)

if (BUILD_TESTS)
    add_subdirectory(Tests)
endif()
