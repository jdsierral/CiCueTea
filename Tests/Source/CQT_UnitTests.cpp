//
//  CQT_UnitTests.cpp
//  CQTDSP_UnitTest
//
//  Created by Juan Sierra on 3/9/25.
//

#include <boost/test/unit_test.hpp>
#include <numbers>
#include <armadillo>
#include <matplot/matplot.h>

#include <CQT.hpp>
#include <VectorOps.h>

using namespace arma;
using namespace std;
using namespace jsa;

namespace plt = matplot;

BOOST_AUTO_TEST_CASE(DFTTest1) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    vec x = ones(fftSize);
    cx_vec X(fftSize/2+1);
    dft.rdft(x, X);
    
    if (false) {
        X.print();
    }
    
    BOOST_CHECK(X[0] == cx_double(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest2) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    vec x = ones(fftSize);
    vec y = zeros(fftSize);
    cx_vec X (fftSize/2+1);
    dft.rdft(x, X);
    dft.irdft(X, y);
    
    if (false) {
        X.print();
    }
    
    BOOST_CHECK(y[0] == 1);
}

BOOST_AUTO_TEST_CASE(DFTTest3) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    cx_vec X = ones<cx_vec>(fftSize);
    cx_vec Y (fftSize);
    dft.dft(X, Y);
    
    if (false) {
        X.print();
    }
    
    BOOST_CHECK(Y[0] == cx_double(fftSize));
}

BOOST_AUTO_TEST_CASE(DFTTest4) {
    DFT dft;
    size_t fftSize = 16;
    dft.init(fftSize);
    cx_vec X = ones<cx_vec>(fftSize);
    cx_vec Y (fftSize);
    dft.idft(X, Y);
    
    if (false) {
        X.print();
    }
    
    BOOST_CHECK(Y[0] == cx_double(1));
}

BOOST_AUTO_TEST_CASE(CQTTestFull1) {
    NsgfCqtFull cqt;
    double fs = 48000;
    int nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    if (false) {
        cout << "nBands: " << cqt.nBands << endl;
        cout << "nFreqs: " << cqt.nFreqs << endl;
        cout << "nSamps: " << cqt.nSamps << endl;
        
        cqt.bax.print();
        plt::figure(1);
        plt::subplot(3, 1, 0);
        for (int k = 0; k < cqt.nBands; k++) {
            vec xi = cqt.fax;
            vec yi = cqt.g.col(k);
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }

        plt::subplot(3, 1, 1);
        plt::semilogx(cqt.fax, cqt.d);

        plt::subplot(3, 1, 2);
        for (int k = 0; k < cqt.nBands; k++) {
            vec xi = cqt.fax;
            vec yi = cqt.gDual.col(k);
            plt::semilogx(cqt.fax, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }
        plt::show();
    }
    
    vec ggDual = sum(cqt.g % cqt.gDual, 1);
    ggDual = ggDual.head_rows(cqt.nFreqs/2-1);
    
    BOOST_CHECK( rms(ggDual - 1) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestFull2) {
    NsgfCqtFull cqt;
    double fs = 48000;
    size_t nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    vec t = regspace(0, nSamps-1) / fs;
    vec x = sin(datum::tau * fRef * t);
    vec y(nSamps);
    cx_mat Xcq(cqt.nSamps, cqt.nBands);
    
    cqt.forward(x, Xcq);
    cqt.inverse(Xcq, y);
    
    if (false) {
        vec Sxx = 10 * log10(sum(square(abs(Xcq)), 1));
        plt::figure();
        plt::semilogx(cqt.bax, Sxx);
        plt::show();
        plt::figure();
        plt::imagesc(toStdVector(20 * log10(abs(Xcq))));
        plt::colorbar();
        plt::show();
        plt::figure();
        plt::subplot(2, 1, 0);
        plt::plot(t, x);
        plt::subplot(2, 1, 1);
        plt::plot(t, y);
        plt::show();
        
        cout << "x: " << rms(x) << endl;
        cout << "y: " << rms(y) << endl;
    }

    vec dif = x - y;
    BOOST_CHECK( rms(dif) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestSparse1) {
    NsgfCqtSparse cqt;
    double fs = 48000;
    int nSamps = 1<<10;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    
    if (false) {
        cout << "nBands: " << cqt.nBands << endl;
        cout << "nFreqs: " << cqt.nFreqs << endl;
        cout << "nSamps: " << cqt.nSamps << endl;
        cqt.bax.print();
        
        plt::figure(1);
        plt::subplot(3, 1, 0);
        for (int k = 0; k < cqt.nBands; k++) {
            uword i0 = cqt.idx[k].i0;
            uword i1 = i0 + cqt.idx[k].len - 1;
            vec xi = cqt.fax.subvec(i0, i1);
            vec yi = cqt.g[k];
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }

        plt::subplot(3, 1, 1);
        plt::semilogx(cqt.fax, cqt.d);

        plt::subplot(3, 1, 2);
        for (int k = 0; k < cqt.nBands; k++) {
            vec xi = cqt.fax.subvec(cqt.idx[k].i0, cqt.idx[k].i0 + cqt.idx[k].len-1);
            vec yi = cqt.gDual[k];
            plt::semilogx(xi, yi);
            plt::ylim({0, 1.5});
            plt::hold(true);
        }
        plt::show();
    }
        
    vec buf = zeros(cqt.nFreqs);
    for (int k = 0; k < cqt.nBands; k++) {
        uword i0 = cqt.idx[k].i0;
        uword len = cqt.idx[k].len;
        uword i1 = i0 + len - 1;
        buf.subvec(i0, i1) += (cqt.g[k] % cqt.gDual[k]);
    }
    buf = buf.head(cqt.nFreqs/2+1);
    buf.save(csv_name("buf.csv"));
    double s = sum(buf - 1);
    BOOST_CHECK( abs(s) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(CQTTestSparse2) {
    NsgfCqtSparse cqt;
    double fs = 48000;
    int nSamps = 1<<16;
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    cqt.init(fs, nSamps, ppo, fMin, fMax, fRef);
    
    vec t = regspace(0, nSamps-1) / fs;
    vec x = sin(datum::tau * fRef * t);
    vec y(nSamps);
    auto Xcq = cqt.getCoefs();
    
    cqt.forward(x, Xcq);
    cqt.inverse(Xcq, y);
    
    if (false) {
//        eig2armaMat(Xcq).save(csv_name("Xcq.csv"));
//        eig2armaMat(cqt.Xmat).save(csv_name("Xmat.csv"));
//        eig2armaVec(cqt.Xdft).save(csv_name("Xdft.csv"));
//        vec Sxx = 10 * (Xcq.abs2().colwise().sum()).log10();
//        plt::figure();
//        plt::semilogx(cqt.bax, Sxx);
//        plt::show();
//        plt::figure();
//        plt::imagesc(toStdVector(20 * Xcq.transpose().abs().log10()));
//        plt::colorbar();
//        plt::show();
        
        plt::figure();
        plt::subplot(2, 1, 0);
        plt::plot(t, x);
        plt::subplot(2, 1, 1);
        plt::plot(t, y);
        plt::show();
        
        cout << "x: " << rms(x) << endl;
        cout << "y: " << rms(y) << endl;
        for (int k = 0; k < cqt.nBands; k++) {
            cout << k << ": " << Xcq[k].size() << endl;
        }
    }
    
    vec dif = x - y;
    BOOST_CHECK( rms(dif) < 1e-10 );
}

BOOST_AUTO_TEST_CASE(SlidingCQT) {
    NsgfCqtFull cqt;
    double fs = 48000;
    uword nSamps = (1<<16);
    double ppo = 1;
    double fMin = 100;
    double fMax = 10000;
    double fRef = 1500;
    
    uword blockSize = 1<<12;
    uword hopSize = blockSize / 2;
    uword overlapSize = blockSize - hopSize;
    uword nBlocks = (nSamps - blockSize) / hopSize;
    vec win = sqrt(hann(blockSize));
    
    cqt.init(fs, blockSize, ppo, fMin, fMax, fRef);
    
    vec t = regspace(0, nSamps-1) / fs;
//    vec x = (2 * M_PI * fRef * t + M_PI_4/2).sin();
    double fScale = 1.2;
    vec x = logChirp(t, fMin * fScale, fMax / fScale);
    x.head(blockSize).fill(0);
    x.tail(blockSize).fill(0);
    x *= hann(nSamps);
    vec y = zeros(nSamps);
    
    vec xi = zeros(blockSize);
    vec xim1 = zeros(blockSize);
    cx_mat Xcq = zeros<cx_mat>(blockSize, cqt.nBands);
    cx_mat Xcqm1 = zeros<cx_mat>(blockSize, cqt.nBands);
    vec yi = zeros(blockSize);
    cx_mat Ycq = zeros<cx_mat>(blockSize, cqt.nBands);
    cx_mat XVal = zeros<cx_mat>(blockSize/2, cqt.nBands);
    cx_mat YVal = zeros<cx_mat>(blockSize/2, cqt.nBands);
    
    cx_mat X = zeros<cx_mat>(nSamps, cqt.nBands); X.fill(0);

    for (uword i = 0; i < nBlocks; i++) {
        uword i0 = i * hopSize;
        uword i1 = i0 + blockSize - 1;
        xi = x.subvec(i0, i1) * win;
        
        cqt.forward(xi, Xcq);
//        XVal = Xcq.topRows(overlapSize) + Xcqm1.bottomRows(overlapSize);
        
        for (int k = 0; k < cqt.nBands; k++) {
//            Ycq.col(k).segment(i0, overlapSize) = Xcqm1.col(k) * win.tail(overlapSize);
//            Ycq.col(k).segment(i0, overlapSize) = XVal.col(k) * win.head(overlapSize);
            X.col(k).subvec(i0, i1) += Xcq.col(k) % win;
        }
        Ycq = Xcq.eval();
        cqt.inverse(Ycq, yi);
        y.subvec(i0, i1) += yi % win;
        
        Xcqm1 = Xcq.eval();
    }
    
//    cout << "Norm: " << X.matrix().norm() << endl;
    
    if (false) {
        //    eig2armaVec(x).save(csv_name("xvec.csv"));
        //    eig2armaMat(X).save(csv_name("Xmat.csv"));
    }
        
    vec d = y - x;
    
    BOOST_CHECK(rms(d) < 1e-10);
}

