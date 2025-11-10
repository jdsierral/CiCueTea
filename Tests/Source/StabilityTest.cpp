//
//  StabilityTest.cpp
//  CiCueTea_UnitTest
//
//  Created by Juan Sierra on 9/1/25.
//

#include <boost/test/unit_test.hpp>

#include <CQT.hpp>
#include <Eigen/Core>
#include "Benchtools.h"

#define N_TESTS 100

using namespace jsa;
using namespace Eigen;

//BOOST_AUTO_TEST_CASE(StabilityTests1)
//{
//    double sampleRate = 48000;
//    Index  nSamps     = 1 << 10;
//    double fraction   = 0.00001;
//    double fMin       = 100;
//    double fMax       = 10000;
//    double fRef       = 1000;
//
//    //    if (NsgfCqtSparse::verifyConfiguration(sampleRate, nSamps, fraction, fMin, fMax, fRef))
//    //        NsgfCqtSparse cqt(sampleRate, nSamps, fraction, fMin, fMax, fRef);
//
//    BOOST_CHECK(true);
//}

BOOST_AUTO_TEST_CASE(BenchTest) {
    
    std::cout << "bench" << std::endl;
}
