//
//  FFTLib_UnitTests.cpp
//  CiCueTea_UnitTest
//
//  Created by Juan Sierra on 8/28/25.
//
//  Benchmarks (CTest label "bench", no correctness assertions): raw timing
//  of the selected FFT backend — rdft across sizes 2^2..2^17, then all four
//  transform directions at 2^16. Correctness of the wrapper is covered by
//  DFT_UnitTests.cpp; this file exists to compare backends and spot
//  performance regressions by eye.
//

#include <FFT.hpp>
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include "Benchtools.h"

#define N_TESTS 100

using namespace jsa::cicuetea;
using namespace jsa::cicuetea::test;
using namespace Eigen;

BOOST_AUTO_TEST_CASE(FFTLibTest1)
{
    std::cout << "Testing " << jsa::cicuetea::DFT::getName() << std::endl;
    for (int j = 2; j < 18; j++) {
        Index    N = 1 << j;
        jsa::cicuetea::DFT dft(N);
        std::cout << "FFT Size: " << N << std::endl;

        {
            Timer t;
            for (int i = 0; i < 10 * N_TESTS; i++) {
                ArrayXd  x = ArrayXd::Random(N);
                ArrayXcd Y(N / 2 + 1);
                dft.rdft(x, Y);
            }
        }
    }

    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(FFTLibTest2)
{
    Index    N = 1 << 16;
    jsa::cicuetea::DFT dft(N);

    {
        Timer t;
        for (int i = 0; i < N_TESTS; i++) {
            ArrayXd  x = ArrayXd::Random(N);
            ArrayXcd Y(N / 2 + 1);
            dft.rdft(x, Y);
        }
    }

    {
        Timer t;
        for (int i = 0; i < N_TESTS; i++) {
            ArrayXcd X = ArrayXcd::Random(N / 2 + 1);
            ArrayXd  y(N);
            dft.irdft(X, y);
        }
    }

    {
        Timer t;
        for (int i = 0; i < N_TESTS; i++) {
            ArrayXcd X = ArrayXcd::Random(N);
            ArrayXcd Y(N);
            dft.dft(X, Y);
        }
    }

    {
        Timer t;
        for (int i = 0; i < N_TESTS; i++) {
            ArrayXcd Y = ArrayXcd::Random(N);
            ArrayXcd X(N);
            dft.idft(Y, X);
        }
    }

    BOOST_CHECK(true);
}
