//
//  Timing_Tests.cpp
//  CiCueTea_UnitTest
//
//  Benchmarks (CTest label "bench", no correctness assertions): wall-time of
//  one full forward + inverse pass over 2^20 samples at 12 bands/octave, for
//  the dense (BenchmarkTest1) and sparse (BenchmarkTest2) transforms —
//  prints the realtime multiple. Correctness round trips live in
//  CQT_UnitTests.cpp.
//

#include <boost/test/unit_test.hpp>

#include <CQT.hpp>
#include <Eigen/Core>
#include <iostream>

#include "Benchtools.h"

using namespace Eigen;
using namespace jsa;

BOOST_AUTO_TEST_CASE(BenchmarkTest1)
{
    double sampleRate = 48000;
    double fs         = sampleRate;
    Index  nSamps     = 1 << 20;
    Index  N          = nSamps;
    double fraction   = 1.0 / 12.0;
    double fMin       = 100;
    double fMax       = 10000;
    double fRef       = 1000;

    jsa::NsgfCqtDense cqt1(sampleRate, nSamps, fraction, fMin, fMax, fRef);

    Index     nBands = cqt1.getNumBands();
    ArrayXd   x      = ArrayXd::Random(nSamps);
    ArrayXd   y      = ArrayXd::Zero(nSamps);
    ArrayXXcd Xcq    = ArrayXXcd::Zero(nSamps, nBands);
    ArrayXXcd Ycq    = ArrayXXcd::Zero(nSamps, nBands);

    Timer tFwd(false);
    cqt1.forward(x, Xcq);
    double dur1 = tFwd.get();

    Ycq = Xcq;

    Timer tInv(false);
    cqt1.inverse(Ycq, y);
    double dur2 = tInv.get();

    double e = sqrt((x - y).square().mean());
    double mul  = (N / fs) * 1000 / (dur1 + dur2);

    std::cout << "Error: " << e << " with duration " << dur1 << "," << dur2 << " ms which is " << mul << "x realtime" << std::endl;
    std::cout << "nBands: " << nBands << std::endl;
    std::cout << Xcq.size() << std::endl;
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(BenchmarkTest2)
{
    double sampleRate = 48000;
    double fs         = sampleRate;
    Index  nSamps     = 1 << 20;
    Index  N          = nSamps;
    double fraction   = 1.0 / 12.0;
    double fMin       = 100;
    double fMax       = 10000;
    double fRef       = 1000;

    jsa::NsgfCqtSparse cqt(sampleRate, nSamps, fraction, fMin, fMax, fRef);

    Index                     nBands = cqt.getNumBands();
    ArrayXd                   x      = ArrayXd::Random(nSamps);
    ArrayXd                   y      = ArrayXd::Zero(nSamps);
    jsa::NsgfCqtSparse::Coefs Xcq    = cqt.getCoefs();

    Timer tFwd(false);
    cqt.forward(x, Xcq);
    double dur1 = tFwd.get();

    Timer tInv(false);
    cqt.inverse(Xcq, y);
    double dur2 = tInv.get();

    double e = sqrt((x - y).square().mean());
    double mul  = (N / fs) * 1000 / (dur1 + dur2);

    std::cout << "Error: " << e << " with duration " << dur1 << "," << dur2 << " ms which is " << mul << "x realtime" << std::endl;
    std::cout << "nBands: " << nBands << std::endl;
    std::cout << Xcq.size() << std::endl;
    BOOST_CHECK(true);
}
