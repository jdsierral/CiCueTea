//
//  SlidingExtended_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 5/6/25.
//

#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <matplot/matplot.h>
#include <numbers>

#include <CQT.hpp>
#include <CQTProcessor.hpp>
#include <SignalUtils.h>
#include <Slicer.hpp>
#include <Splicer.hpp>

#include "VectorOps.h"

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

#include "EmptyCQTProc.h"

BOOST_AUTO_TEST_CASE(ExtendedTest1) {
    std::cout << BOOST_CURRENT_LOCATION;
    return;
    double fs = 48000;
    Index N = exp2(18);
    Index blockSize = exp2(16);
    
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);
    
    SliCQTFull ola(fs, blockSize, 12, 1e2, 1e4, 1e3);
    
    eig2armaVec(ola.getCqt().NsgfCqtCommon::getFrequencyAxis()).save(csv_name("fax.csv"));
    eig2armaVec(ola.getWindow()).save(csv_name("win.csv"));
    eig2armaVec(ola.getCqt().getDiagonalization()).save(csv_name("d.csv"));
    for (int k = 0; k < ola.getCqt().getNumBands(); k++) {
        std::string name = "g" + std::to_string(k+1) + ".csv";
        std::string dualName = "gDual" + std::to_string(k+1) + ".csv";
    }

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }
    
    eig2armaVec(x).save(csv_name("x.csv"));
    eig2armaVec(y).save(csv_name("y.csv"));
    
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(ExtendedTest2) {
    std::cout << BOOST_CURRENT_LOCATION;
    double fs = 48000;
    Index N = exp2(18);
    Index blockSize = exp2(16);
    
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);
    double frac = 1;
    double fMin = 1e2;
    double fMax = 1e4;
    double fRef = 1e3;
    
    SliCQTSparse ola(fs, blockSize, frac, fMin, fMax, fRef);
    
    eig2armaVec(ola.getCqt().NsgfCqtCommon::getFrequencyAxis()).save(csv_name("fax.csv"));
    eig2armaVec(ola.getWindow()).save(csv_name("win.csv"));
    eig2armaVec(ola.getCqt().getDiagonalization()).save(csv_name("d.csv"));
    for (int k = 0; k < ola.getCqt().getNumBands(); k++) {
        std::string name = "g" + std::to_string(k+1) + ".csv";
        std::string dualName = "gDual" + std::to_string(k+1) + ".csv";
        eig2armaVec(ola.getCqt().getAtom(k)).save(csv_name(name));
        eig2armaVec(ola.getCqt().getDualAtom(k)).save(csv_name(dualName));
    }

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }
    
    eig2armaVec(x).save(csv_name("x.csv"));
    eig2armaVec(y).save(csv_name("y.csv"));
    
    BOOST_TEST(true);
}
