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

#define SIG_LEN exp2(18)
#define BLOCK_SIZE exp2(16)
#define FRAC 1.0 / 12.0
#define FMIN 1e2
#define FMAX 1e4
#define FREF 1e3

BOOST_AUTO_TEST_CASE(DoubleBufferTest)
{
    //    DoubleBuffer<double> buffer;
    //    double zero = 0;
    //    double one = 1;
    //    double two = 2;
    //    buffer.fill(zero);
    //    buffer.print();
    //    buffer.push(one);
    //    buffer.print();
    //    buffer.push(two);
    //    buffer.print();
    //    double val1 = buffer.next();
    //    double val2 = buffer.current();
    //    std::cout << val1 << val2 << std::endl;
    //    buffer.push(one);
    //    buffer.print();
    //    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(ExtendedTest1)
{
    return;
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  N         = SIG_LEN;
    Index  blockSize = BLOCK_SIZE;
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    CqtFull ola(fs, blockSize, frac, fMin, fMax, fRef);
    string  baseName = "CqtFull_";
    eig2armaVec(ola.getCqt().NsgfCqtCommon::getFrequencyAxis()).save(csv_name(baseName + "fax.csv"));
    eig2armaVec(ola.getWindow()).save(csv_name(baseName + "win.csv"));
    eig2armaVec(ola.getCqt().getDiagonalization()).save(csv_name(baseName + "d.csv"));
    for (int k = 0; k < ola.getCqt().getNumBands(); k++) {
        std::string name     = baseName + "g" + std::to_string(k + 1) + ".csv";
        std::string dualName = baseName + "gDual" + std::to_string(k + 1) + ".csv";
    }

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();

    x = x.head(N - latency);
    y = y.tail(N - latency);

    eig2armaVec(x).save(csv_name(baseName + "x.csv"));
    eig2armaVec(y).save(csv_name(baseName + "y.csv"));

    double err = rms(y - x);
    std::cout << err << std::endl;
    BOOST_TEST(err < 0.1);
}

BOOST_AUTO_TEST_CASE(ExtendedTest2)
{
    return;
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  N         = SIG_LEN;
    Index  blockSize = BLOCK_SIZE;
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    CqtSparse ola(fs, blockSize, frac, fMin, fMax, fRef);
    string    baseName = "CqtSparse_";

    eig2armaVec(ola.getCqt().NsgfCqtCommon::getFrequencyAxis()).save(csv_name(baseName + "fax.csv"));
    eig2armaVec(ola.getWindow()).save(csv_name(baseName + "win.csv"));
    eig2armaVec(ola.getCqt().getDiagonalization()).save(csv_name(baseName + "d.csv"));
    for (int k = 0; k < ola.getCqt().getNumBands(); k++) {
        std::string name     = baseName + "g" + std::to_string(k + 1) + ".csv";
        std::string dualName = baseName + "gDual" + std::to_string(k + 1) + ".csv";
    }

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();

    x = x.head(N - latency);
    y = y.tail(N - latency);

    eig2armaVec(x).save(csv_name(baseName + "x.csv"));
    eig2armaVec(y).save(csv_name(baseName + "y.csv"));

    double err = rms(y - x);
    std::cout << err << std::endl;
    BOOST_TEST(err < 0.1);
}

BOOST_AUTO_TEST_CASE(ExtendedTest3)
{
    return;
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  N         = SIG_LEN;
    Index  blockSize = BLOCK_SIZE;
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    ArrayXd x = ArrayXd::Random(N);
    x         = sin(2 * M_PI * 1000 * regspace(N) / fs);
    ArrayXd y = ArrayXd::Zero(N);

    SliCQTFull ola(fs, blockSize, frac, fMin, fMax, fRef);
    string     baseName = "SliCQTFull_";
    eig2armaVec(ola.getCqt().NsgfCqtCommon::getFrequencyAxis()).save(csv_name(baseName + "fax.csv"));
    eig2armaVec(ola.getWindow()).save(csv_name(baseName + "win.csv"));
    eig2armaVec(ola.getCqt().getDiagonalization()).save(csv_name(baseName + "d.csv"));
    for (int k = 0; k < ola.getCqt().getNumBands(); k++) {
        std::string name     = baseName + "g" + std::to_string(k + 1) + ".csv";
        std::string dualName = baseName + "gDual" + std::to_string(k + 1) + ".csv";
    }

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();
    latency       = ola.getCqt().getBlockSize();

    x = x.head(N - latency);
    y = y.tail(N - latency);

    eig2armaVec(x).save(csv_name(baseName + "x.csv"));
    eig2armaVec(y).save(csv_name(baseName + "y.csv"));

    double err = rms(y - x);
    std::cout << err << std::endl;
    BOOST_TEST(err < 0.1);
}

BOOST_AUTO_TEST_CASE(ExtendedTest4)
{
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  N         = SIG_LEN;
    Index  blockSize = BLOCK_SIZE;
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    SliCQTSparse ola(fs, blockSize, frac, fMin, fMax, fRef);
    string       baseName = "";
    eig2armaVec(ola.getCqt().NsgfCqtCommon::getFrequencyAxis()).save(csv_name(baseName + "fax.csv"));
    eig2armaVec(ola.getWindow()).save(csv_name(baseName + "win.csv"));
    eig2armaVec(ola.getCqt().getDiagonalization()).save(csv_name(baseName + "d.csv"));
    for (int k = 0; k < ola.getCqt().getNumBands(); k++) {
        std::string name     = baseName + "g" + std::to_string(k + 1) + ".csv";
        std::string dualName = baseName + "gDual" + std::to_string(k + 1) + ".csv";
        std::string winName  = baseName + "Win" + std::to_string(k + 1) + ".csv";
        eig2armaVec(ola.getCqt().getAtom(k)).save(csv_name(name));
        eig2armaVec(ola.getCqt().getDualAtom(k)).save(csv_name(dualName));
        eig2armaVec(ola.getCqtWindow(k)).save(csv_name(winName));
    }

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();

    x = x.head(N - latency);
    y = y.tail(N - latency);

    eig2armaVec(x).save(csv_name(baseName + "x.csv"));
    eig2armaVec(y).save(csv_name(baseName + "y.csv"));

    double err = rms(y - x);
    std::cout << err << std::endl;
    BOOST_TEST(err < 0.1);
}
