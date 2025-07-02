cmake_minimum_required(VERSION 3.25)

execute_process(COMMAND sh -c "git rev-parse --short HEAD > ${CMAKE_CURRENT_SOURCE_DIR}/GIT_COMMIT")
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION CURRENT_VERSION)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/GIT_COMMIT CURRENT_GIT_COMMIT)

project("CQTDSP" VERSION ${CURRENT_VERSION})

set(TARGET_NAME ${PROJECT_NAME})

set(CHOC_INCLUDE_DIRS /Users/juans/Developer/DevTools/choc)
set(CMAKE_INCLUDE_PATH /usr/local/include)

find_package(Armadillo REQUIRED)
cmake_policy(SET CMP0167 NEW)
find_package(Boost 1.82 REQUIRED)
find_package(Boost COMPONENTS program_options log log_setup REQUIRED)
find_package(JUCE REQUIRED)
find_package(Matplot++)

set(ARMADILLO_CONFIG
    # ARMA_NO_DEBUG
    ARMA_BLAS_LONG
    ARMA_DONT_USE_STD_MUTEX
    # ARMA_DONT_PRINT_EXCEPTIONS
    ARMA_MAT_PREALLOC=32
    # ARMA_DONT_ZERO_INIT
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64
    ARMA_DONT_USE_WRAPPER
    ARMA_BLAS_CAPITALS
)


include(SourceFiles.cmake)

add_library(${TARGET_NAME} STATIC ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        # -Weverything
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE ${ARMADILLO_CONFIG}
)

target_include_directories(${TARGET_NAME}
    PRIVATE Include/
    SYSTEM PRIVATE ${CMAKE_INCLUDE_PATH} 
    SYSTEM PRIVATE ${Boost_INCLUDE_DIRS}
    SYSTEM PRIVATE ${ARMADILLO_INCLUDE_DIRS}
)

set_target_properties(${TARGET_NAME} 
    PROPERTIES 
        XCODE_GENERATE_SCHEME ON
)

add_subdirectory(Tests)