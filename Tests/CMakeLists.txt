enable_testing()

set(TARGET_NAME ${PROJECT_NAME}_UnitTest)

set(SourceFiles
    Source/main.cpp
    Source/Perf_UnitTests.cpp
)

source_group("Source Files\\Source" FILES ${SourceFiles})

add_executable(${TARGET_NAME})
target_sources(${TARGET_NAME} PRIVATE ${SourceFiles})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

target_compile_definitions(${TARGET_NAME}
    PRIVATE ${ARMADILLO_CONFIG}
    PRIVATE EIGEN_RUNTIME_NO_MALLOC
    PRIVATE EIGEN_USE_BLAS
    PRIVATE EIGEN_USE_LAPACKE
)


target_compile_options(${TARGET_NAME}
    PRIVATE -fcolor-diagnostics
)

target_include_directories(${TARGET_NAME}
    PRIVATE ${ARMADILLO_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_INCLUDE_PATH}
    PRIVATE ../Source
)

target_link_libraries(${TARGET_NAME}
    PRIVATE "-framework Accelerate"
    PRIVATE Matplot++::matplot
    PRIVATE ${PROJECT_NAME}
    PRIVATE Boost::log_setup Boost::log
)

# add_test(Test1 LibraryUnitTest COMMAND ${TARGET_NAME})

set_target_properties(${TARGET_NAME} 
    PROPERTIES
        XCODE_GENERATE_SCHEME ON
        XCODE_SCHEME_ARGUMENTS --color_output=no
        XCODE_SCHEME_ENVIRONMENT "PATH=/opt/homebrew/bin"
)