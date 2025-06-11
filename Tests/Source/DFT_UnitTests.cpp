//
//  DFT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 4/8/25.
//

#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <matplot/matplot.h>
#include <numbers>

#include <CQT.hpp>
#include <VectorOps.h>

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

namespace plt = matplot;

#include <fftw3.h>

#define FFT_SIZE 32
#define SHOULD_PRINT false

BOOST_AUTO_TEST_CASE(DFTMultiTest1) {
    int rank = 1;
    const int N = 1<<6;
    int howmany = 10;
    ArrayXXd x = ArrayXXd::Ones(N, howmany);
    int istride = 1;
    int idist = N;
    ArrayXXcd X = ArrayXXcd::Zero(N/2+1, howmany);
    int ostride = 1;
    int odist = N/2+1;
    
    
    fftw_plan plan = fftw_plan_many_dft_r2c(rank, &N, howmany,
                                            x.data(), nullptr, istride, idist,
                                            (fftw_complex*)X.data(), nullptr, ostride, odist,
                                            FFTW_MEASURE);
    
    fftw_execute(plan);
    //    fftw_execute_dft_r2c(plan, x.data(), (fftw_complex*)X.data());
    
    eig2armaMat(x).print();
    eig2armaMat(X).print();
    
    
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(DFTTest1) {
    size_t fftSize = FFT_SIZE;
    DFT dft(fftSize);
    ArrayXd x = ArrayXd::Ones(fftSize);
    ArrayXcd X(fftSize / 2 + 1);
    dft.rdft(x, X);
    
    if (SHOULD_PRINT) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(X[0] == dcomplex(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest2) {
    size_t fftSize = FFT_SIZE;
    DFT dft(fftSize);
    ArrayXd x = ArrayXd::Ones(fftSize);
    ArrayXcd X(fftSize / 2 + 1);
    ArrayXd y = ArrayXd::Zero(fftSize);
    dft.rdft(x, X);
    dft.irdft(X, y);
    
    if (SHOULD_PRINT) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(X[0] == dcomplex(fftSize));
    BOOST_CHECK(y[0] == 1);
}

BOOST_AUTO_TEST_CASE(DFTTest3) {
    size_t fftSize = FFT_SIZE;
    DFT dft(fftSize);
    ArrayXcd X = ArrayXcd::Ones(fftSize);
    ArrayXcd Y(fftSize);
    dft.dft(X, Y);
    
    if (SHOULD_PRINT) {
        eig2armaVec(X).print();
    }
    
    BOOST_CHECK(Y[0] == dcomplex(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest4) {
    size_t fftSize = FFT_SIZE;
    DFT dft(fftSize);
    ArrayXcd X = ArrayXcd::Ones(fftSize);
    ArrayXcd Y(fftSize);
    dft.idft(X, Y);
    
    if (SHOULD_PRINT) {
        eig2armaVec(Y).print();
    }
    
    BOOST_CHECK(Y[0] == dcomplex(1));
}
