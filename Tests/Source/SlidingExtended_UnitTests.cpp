//
//  SlidingExtended_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 5/6/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>

#include <CQT.hpp>
#include <CQTProcessor.hpp>
#include <SignalUtils.h>
#include <Slicer.hpp>
#include <Splicer.hpp>

#include "VectorOps.h"

using namespace Eigen;
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

    CqtDense ola(fs, blockSize, frac, fMin, fMax, fRef);
    string   baseName = "CqtDense_";

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();

    x = x.head(N - latency);
    y = y.tail(N - latency);


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


    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();

    x = x.head(N - latency);
    y = y.tail(N - latency);


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

    SliCqtDense ola(fs, blockSize, frac, fMin, fMax, fRef);
    string      baseName = "SliCQTDense_";

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();
    latency       = ola.getCqt().getBlockSize();

    x = x.head(N - latency);
    y = y.tail(N - latency);


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

    SliCqtSparse ola(fs, blockSize, frac, fMin, fMax, fRef);
    string       baseName = "";

    for (int n = 0; n < N; n++) {
        y[n] = ola.processSample(x[n]);
    }

    Index latency = ola.getLatency();

    x = x.head(N - latency);
    y = y.tail(N - latency);


    double err = rms(y - x);
    std::cout << err << std::endl;
    BOOST_TEST(err < 0.1);
}
