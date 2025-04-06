//
//  SlidingCQT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 3/23/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>
#include <armadillo>
#include <matplot/matplot.h>

#include <CQT.hpp>
#include <OverlapAddProcessor.hpp>
#include <Splicer.hpp>
#include <Slicer.hpp>
#include <OverlapAddProcessor.hpp>
#include <VectorOps.h>

using namespace arma;
using namespace std;
using namespace jsa;

class cqtFull : public jsa::cqtFullProcessor
{
public:
    void processBlock(arma::cx_mat& block) override {}
};

class sliCQTFull : public jsa::slidingCQTFullProcessor
{
public:
    void processBlock(arma::cx_mat& block) override {};
};

class cqtSparse : public jsa::cqtSparseProcessor
{
public:
    void processBlock(jsa::NsgfCqtSparse::Coefs& block) override {}
};

class sliCQTSparse : public jsa::slidingCqtSparseProcessor
{
public:
    void processBlock(jsa::NsgfCqtSparse::Coefs& block) override {};
};


BOOST_AUTO_TEST_CASE(OlaProc1) {
    double fs = 48000;
    uword N = 1<<10;
    uword blockSize = 1<<5;
    uword overlapSize = 1<<4;
    
    vec x = randn(N);
    vec y = zeros(N);
    
    cqtFull ola;
    ola.init(fs, blockSize, 1, 1e2, 1e4, 1e3);
    
    for (uword n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }
    
    vec d = x.head(N-blockSize) - y.tail(N-blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(OlaProc2) {
    double fs = 48000;
    uword N = 1<<20;
    uword blockSize = 1<<16;
    
    vec x = randn(N);
    vec y = zeros(N);
    
    sliCQTFull ola;
    ola.init(fs, blockSize, 1, 1e2, 1e4, 1e3);
    
    for (uword n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }
    
    x = x.head(N-blockSize-blockSize/2);
    y = y.tail(N-blockSize-blockSize/2);
    
    vec d = x - y;
//    cout << rms(d) << endl;
//    BOOST_CHECK(rms(d) < 1e-4);
}

BOOST_AUTO_TEST_CASE(OlaProc3) {
    double fs = 48000;
    uword N = 1<<16;
    uword blockSize = 1<<10;
    uword overlapSize = blockSize/2;
    
    vec x = randn(N);
    vec y = zeros(N);
    
    cqtSparse ola;
    ola.init(fs, blockSize, 1, 1e2, 1e4, 1e3);
    
    for (uword n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }
    
    vec d = x.head(N-blockSize) - y.tail(N-blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(OlaProc4) {
    double fs = 48000;
    uword N = 1<<18;
    uword blockSize = 1<<16;
    double ppo = 3;
    double fMax = 1e4;
    double fMin = 1e2;
    double fRef = 1e3;
    
    vec x = randn(N);
    vec y = zeros(N);
    
    sliCQTSparse ola;
    ola.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
    for (uword n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }
    
    x = x.head(N-blockSize-blockSize/2);
    y = y.tail(N-blockSize-blockSize/2);
    
    vec d = x - y;
//    cout << rms(d) << endl;
//    BOOST_CHECK(rms(d) < 1e-4);
}

