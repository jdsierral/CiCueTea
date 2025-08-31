//
//  MultiDFT_UnitTests.cpp
//  CiCueTea_UnitTest
//
//  Created by Juan Sierra on 8/28/25.
//

#include <armadillo>
#include <boost/test/unit_test.hpp>
#include <matplot/matplot.h>
#include <numbers>

#include <CQT.hpp>

#include "VectorOps.h"

using namespace Eigen;
using namespace arma;
using namespace std;
using namespace jsa;


#include <fftw3.h>

BOOST_AUTO_TEST_CASE(DFTMultiTest1)
{
    int       rank    = 1;
    int       N       = 1 << 6;
    int       howmany = 10;
    ArrayXXd  x       = ArrayXXd::Ones(N, howmany);
    int       istride = 1;
    int       idist   = N;
    ArrayXXcd X       = ArrayXXcd::Zero(N / 2 + 1, howmany);
    int       ostride = 1;
    int       odist   = N / 2 + 1;

    fftw_plan plan = fftw_plan_many_dft_r2c(
        rank, &N, howmany, x.data(), nullptr, istride, idist,
        (fftw_complex*)X.data(), nullptr, ostride, odist, FFTW_MEASURE);

    fftw_execute(plan);
    //    fftw_execute_dft_r2c(plan, x.data(), (fftw_complex*)X.data());

    //  eig2armaMat(x).print();
    //  eig2armaMat(X).print();

    BOOST_TEST(true);
}
