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

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

class Ola : public jsa::NsgfCqtProcessor
{
public:
    void processBlock(Eigen::ArrayXXcd& block) override {}
};

class FullOla : public jsa::SliCQOlaProcessor
{
public:
    void processBlock(Eigen::ArrayXXcd& block) override {};
};

BOOST_AUTO_TEST_CASE(OlaProc1) {
    double fs = 48000;
    Index N = 1<<10;
    Index blockSize = 1<<5;
    Index overlapSize = 1<<4;
    
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);
    
    Ola ola;
    ola.init(fs, blockSize, 1, 1e4, 1e2, 1e3);
    
    for (Index n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }
    
    ArrayXd d = x.head(N-blockSize) - y.tail(N-blockSize);
    BOOST_CHECK(rms(d) < 1e-10);
}

BOOST_AUTO_TEST_CASE(OlaProc2) {
    double fs = 48000;
    Index N = 1<<20;
    Index blockSize = 1<<16;
    
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);
    
    FullOla ola;
    ola.init(fs, blockSize, 1, 1e4, 1e2, 1e3);
    
    for (Index n = 0; n < N; n++) {
        y(n) = ola.processSample(x(n));
    }
    
    x = x.head(N-blockSize-blockSize/2);
    y = y.tail(N-blockSize-blockSize/2);
    jsa::eig2armaVec(x).save(csv_name("x.csv"));
    jsa::eig2armaVec(y).save(csv_name("y.csv"));
    
    ArrayXd d = x - y;
    cout << rms(d) << endl;
    BOOST_CHECK(rms(d) < 1e-5);
}

