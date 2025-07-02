//
//  CQT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 3/9/25.
//

#include <boost/test/unit_test.hpp>
#include <CQT.hpp>

#include <matplot/matplot.h>

using namespace arma;

BOOST_AUTO_TEST_CASE(DFTTest1) {
    jsa::DFT dft;
    uword fftSize = 1<<4;
    dft.init(fftSize);
    
    vec x = ones(fftSize);
    cx_vec X = zeros<cx_vec>(fftSize/2+1);
    dft.rdft(x, X);
    X.print();
    BOOST_CHECK(X[0] == 1.0);
}

BOOST_AUTO_TEST_CASE(CQTTestFull) {
    jsa::NsgfCqtFull cqt;
    double fs = 48000;
    size_t nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1000;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
        
    vec t = regspace(0, nSamps) / fs;
    vec x = sin(datum::tau * fRef * t);
    vec y(nSamps);
    cx_mat X(cqt.nSamps, cqt.nBands);
    
    cqt.forward(x, X);
    cqt.inverse(X, y);
    
//    matplot::figure();
//    matplot::subplot(2, 1, 0);
//    matplot::plot(t, x);
//    matplot::subplot(2, 1, 1);
//    matplot::plot(t, y);
//    matplot::show();
    
    BOOST_CHECK(approx_equal(x, y, "absdiff", 1e-10));
}
