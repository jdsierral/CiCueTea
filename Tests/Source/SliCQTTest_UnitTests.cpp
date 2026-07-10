//
//  SliCQTTest_UnitTests.cpp
//  SliCQTTest_UnitTests
//
//  Created by Juan Sierra on 3/9/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>

#include <CQT.hpp>
#include <MathUtils.h>
#include <SignalUtils.h>

#include "VectorOps.h"

using namespace Eigen;
using namespace std;
using namespace jsa;

#include "CQTProcessor.hpp"
#include "EmptyCQTProc.h"

#define SIG_LEN exp2(15)
#define BLOCK_SIZE exp2(16)
#define FRAC 1.0 / 12.0
#define FMIN 1e2
#define FMAX 1e4
#define FREF 1e3

BOOST_AUTO_TEST_CASE(SparseCQTPhaseTest1)
{
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  blockSize = exp2(16);
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    NsgfCqtSparse cqt(fs, blockSize, frac, fMin, fMax, fRef);

    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(SparseCQTTest1)
{
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  blockSize = exp2(16);
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    NsgfCqtSparse        cqt(fs, blockSize, frac, fMin, fMax, fRef);
    ArrayXd              x  = ArrayXd::Random(blockSize);
    ArrayXd              y  = ArrayXd::Zero(blockSize);
    NsgfCqtSparse::Coefs Xi = cqt.getCoefs();

    cqt.forward(x, Xi);
    cqt.inverse(Xi, y);

    double err = rms(x - y);
    BOOST_CHECK(err < 0.1);
}

BOOST_AUTO_TEST_CASE(SliCQTTest1)
{
    std::cout << BOOST_CURRENT_LOCATION << std::endl;
    double fs        = 48000;
    Index  N         = exp2(16);
    Index  blockSize = exp2(16);
    double frac      = FRAC;
    double fMin      = FMIN;
    double fMax      = FMAX;
    double fRef      = FREF;

    ArrayXd x = ArrayXd::Random(N);
    ArrayXd y = ArrayXd::Zero(N);

    SliCqtSparse ola(fs, blockSize, frac, fMin, fMax, fRef);

    for (int i = 0; i < N; i++) {
        y[i] = ola.processSample(x[i]);
    }

    BOOST_CHECK(true);
}
