//
//  SlidingExtended_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 5/6/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>
#include <armadillo>
#include <matplot/matplot.h>

#include <CQT.hpp>
#include <CQTProcessor.hpp>
#include <RingBuffer.hpp>
#include <Splicer.hpp>
#include <Slicer.hpp>
#include <VectorOps.h>
#include <SignalUtils.h>

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;

#include "EmptyCQTProc.h"

BOOST_AUTO_TEST_CASE(ExtendedTest1) {
    double fs = 48000;
    Index N = exp2(18);
    Index blockSize = exp2(12);
    
    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);
    
    sliCQTFull ola(fs, blockSize, 1, 1e2, 1e4, 1e3);
    
//    eig2armaVec(ola.cqt.NsgfCqtCommon::getFrequencyAxis()).save(csv_name("fax.csv"));
//    eig2armaVec(ola.getWindow()).save(csv_name("win.csv"));
//    eig2armaVec(ola.cqt.getDiagonalization()).save(csv_name("d.csv"));
//    for (int k = 0; k < ola.cqt.getNumBands(); k++) {
//        std::string name = "g" + std::to_string(k+1) + ".csv";
//        std::string dualName = "gDual" + std::to_string(k+1) + ".csv";
//        eig2armaVec( ola.cqt.getAtom(k) ).save(csv_name(name));
//        eig2armaVec( ola.cqt.getDualAtom(k) ).save(csv_name(dualName));
//    }
//    
//    for (int n = 0; n < N; n++) {
//        y[n] = ola.processSample(x[n]);
//    }
    
    eig2armaVec(x).save(csv_name("x.csv"));
    eig2armaVec(y).save(csv_name("y.csv"));
    
    BOOST_TEST(true);
}

