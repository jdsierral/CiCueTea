//
//  FFTLib_UnitTests.cpp
//  CiCueTea_UnitTest
//
//  Created by Juan Sierra on 8/28/25.
//

#include <FFT.hpp>
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include "Benchtools.h"

#define N_TESTS 100

using namespace jsa;
using namespace Eigen;

BOOST_AUTO_TEST_CASE(FFTLibTest1)
{
    std::cout << "Testing " << jsa::DFT::getName() << std::endl;
    for (int j = 2; j < 18; j++) {
        Index    N = 1 << j;
        jsa::DFT dft(N);
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
    jsa::DFT dft(N);

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
