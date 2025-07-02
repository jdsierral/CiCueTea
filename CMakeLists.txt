cmake_minimum_required(VERSION 4.0)
include(GitData.cmake)
project("CiCueTea" VERSION ${CURRENT_VERSION})
message(STATUS ${SEP} ${PROJECT_NAME} ${SEP})

set(TARGET_NAME ${PROJECT_NAME})

find_package(Eigen3 3.3 REQUIRED)
cmake_policy(SET CMP0167 NEW)
find_package(Boost 1.82 REQUIRED)
find_package(Boost COMPONENTS program_options log log_setup REQUIRED)
find_package(Matplot++)
option(BUILD_TESTS "Build all the unit tests" OFF)

include(SourceFiles.cmake)
include(FFTSelection.cmake)

add_library(${TARGET_NAME} STATIC ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        # -Weverything
)

target_compile_definitions(${TARGET_NAME}
    PRIVATE 
        EIGEN_RUNTIME_NO_MALLOC
        ${FFT_NAME}
)

target_include_directories(${TARGET_NAME}
    PRIVATE Include/
    SYSTEM PRIVATE /usr/local/include
    SYSTEM PRIVATE ${CMAKE_INCLUDE_PATH} 
    SYSTEM PRIVATE ${Boost_INCLUDE_DIRS}
    SYSTEM PRIVATE ${FFT_INCLUDE_DIR}
)

set_target_properties(${TARGET_NAME} 
    PROPERTIES 
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_LAUNCH_CONFIGURATION "Release"
)

target_link_libraries(${TARGET_NAME} 
    PRIVATE Eigen3::Eigen
    PRIVATE ${FFT_LINK_CMD}
)

if (BULD_TESTS)
    add_subdirectory(Tests)
endif()