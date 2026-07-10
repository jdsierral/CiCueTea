//
//  DFT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 4/8/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>

#include <CQT.hpp>

#include "VectorOps.h"

using namespace Eigen;
using namespace std;
using namespace jsa;

#define FFT_SIZE 32

BOOST_AUTO_TEST_CASE(DFTTest1)
{
    size_t   fftSize = FFT_SIZE;
    DFT      dft(fftSize);
    ArrayXd  x = ArrayXd::Ones(fftSize);
    ArrayXcd X(fftSize / 2 + 1);
    dft.rdft(x, X);


    BOOST_CHECK(X[0] == complex<double>(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest2)
{
    size_t   fftSize = FFT_SIZE;
    DFT      dft(fftSize);
    ArrayXd  x = ArrayXd::Ones(fftSize);
    ArrayXcd X(fftSize / 2 + 1);
    ArrayXd  y = ArrayXd::Zero(fftSize);
    dft.rdft(x, X);
    dft.irdft(X, y);


    BOOST_CHECK(X[0] == complex<double>(fftSize));
    BOOST_CHECK(y[0] == 1);
}

BOOST_AUTO_TEST_CASE(DFTTest3)
{
    size_t   fftSize = FFT_SIZE;
    DFT      dft(fftSize);
    ArrayXcd X = ArrayXcd::Ones(fftSize);
    ArrayXcd Y(fftSize);
    dft.dft(X, Y);


    BOOST_CHECK(Y[0] == complex<double>(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest4)
{
    size_t   fftSize = FFT_SIZE;
    DFT      dft(fftSize);
    ArrayXcd X = ArrayXcd::Ones(fftSize);
    ArrayXcd Y(fftSize);
    dft.idft(X, Y);


    BOOST_CHECK(Y[0] == complex<double>(1));
}
