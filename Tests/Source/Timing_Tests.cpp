#include <boost/test/unit_test.hpp>

#include <CQT.hpp>
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;

BOOST_AUTO_TEST_CASE(BenchmarkTest1)
{
    double sampleRate = 48000;
    double fs = sampleRate;
    Index  nSamps     = 1 << 20;
    Index N = nSamps;
    double fraction   = 1.0/12.0;
    double fMin       = 100;
    double fMax       = 10000;
    double fRef       = 1000;

    jsa::NsgfCqtDense cqt1(sampleRate, nSamps, fraction, fMin, fMax, fRef);

    Index nBands = cqt1.getNumBands();
    ArrayXd x = ArrayXd::Random(nSamps);
    ArrayXd y = ArrayXd::Zero(nSamps);
    ArrayXXcd Xcq = ArrayXXcd::Zero(nSamps, nBands);
    ArrayXXcd Ycq = ArrayXXcd::Zero(nSamps, nBands);

    auto t0 = std::chrono::steady_clock::now();
    cqt1.forward(x, Xcq);
    auto t1 = std::chrono::steady_clock::now();
    Ycq = Xcq;
    auto t2 = std::chrono::steady_clock::now();
    cqt1.inverse(Ycq, y);
    auto t3 = std::chrono::steady_clock::now();

    double e = sqrt((x-y).square().mean());
    auto dur1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto dur2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double mul = (N  / fs) * 1000 / (dur1 + dur2);

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

    auto t0 = std::chrono::steady_clock::now();
    cqt.forward(x, Xcq);
    auto t1 = std::chrono::steady_clock::now();
    auto t2 = std::chrono::steady_clock::now();
    cqt.inverse(Xcq, y);
    auto t3 = std::chrono::steady_clock::now();

    double e    = sqrt((x - y).square().mean());
    auto   dur1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto   dur2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double mul  = (N / fs) * 1000 / (dur1 + dur2);

    std::cout << "Error: " << e << " with duration " << dur1 << "," << dur2 << " ms which is " << mul << "x realtime" << std::endl;
    std::cout << "nBands: " << nBands << std::endl;
    std::cout << Xcq.size() << std::endl;
    BOOST_CHECK(true);
}
