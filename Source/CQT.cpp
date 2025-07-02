//
//  CQT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "CQT.hpp"

#include <cassert>

using namespace jsa;
using namespace arma;

inline double square(double x) { return x * x; }

void NsgfCqtCommon::init(double sampleRate, size_t numSamples, double pointsPerOctave,
                         double minFrequency, double maxFrequency, double refFrequency)
{
    fs = sampleRate;
    nSamps = numSamples;
    ppo = pointsPerOctave;
    fMin = minFrequency;
    fMax = maxFrequency;
    fRef = refFrequency;
    
    // Initialize Stuff
    X.resize(nSamps);
    Y.resize(nSamps);
    dft.init(nSamps);
    
    int nBandsUp = int(ceil(ppo * log2(fMax / fRef)));
    int nBandsDown = int(ceil(ppo * log2(fRef / fMin)));
    bax = fRef * exp(log(2) * regspace(-nBandsDown, nBandsUp) / ppo);
    nBands = bax.size();
    
    nFreqs = nSamps;
    fax = regspace<vec>(0, nFreqs-1) * fs / double(nFreqs);
}

//==========================================================================
//==========================================================================
//==========================================================================

void NsgfCqtFull::init(double sampleRate, size_t numSamples, double ppo,
                       double minFrequency, double maxFrequency, double refFrequency)
{
    
    NsgfCqtCommon::init(sampleRate, numSamples, ppo,
                        minFrequency, maxFrequency, refFrequency);
    
    Xi.resize(nSamps, nBands);
    Yi.resize(nSamps, nBands);
    
    double c = log(16) / (square(1.0 / ppo));
    mat outerRatio = fax * (1.0 / bax.t());
    g = exp(-c * square(log2(outerRatio)));
    
    uword end = nBands - 1;
    
    for (uword k = 0; k < fax.size(); k++) {
        if (fax(k) <  bax(0) ) g(k,  0 ) = 1.0; // g( 0 , find(fax < bax( 1 ))).ones(1);
        if (fax(k) > bax(end)) g(k, end) = 1.0; // g(end, find(fax > bax(end))).ones(1);
        if (fax(k) < fs/2) continue;
        for (uword b = 0; b < nBands; b++) {
            g(k,b) = 0;
        }
    }
    
    d = sum(arma::square(g), 1);
    gDual = g.each_col() / d;
}

void NsgfCqtFull::forward(const vec& x, cx_mat& Xcq) {
    assert(Xcq.n_cols == nBands);
    assert(Xcq.n_rows == nSamps);
    dft.rdft(x, X);
    for (uword k = 0; k < nBands; k++) {
        Xi.col(k) = g.col(k) % X;
    }
    dft.idft(Xi, Xcq);
}

void NsgfCqtFull::inverse(const cx_mat& Xcq, vec& x) {
    dft.dft(Xcq, Yi);
    Y = sum(Yi, 1);
    dft.irdft(Y, x);
}


//==========================================================================
//==========================================================================
//==========================================================================

void NsgfCqtSparse::init(double sampleRate, size_t numSamples, double ppo,
                         double minFrequency, double maxFrequency, double refFrequency)
{
    NsgfCqtCommon::init(sampleRate, numSamples, ppo,
                        minFrequency, maxFrequency, refFrequency);
    
    // Initialize Stuff
//    idx.resize(nBands);
//    g.resize(nBands);
//    gDual.resize(nBands);
//    phase.resize(nBands);
//    scale.resize(nBands);
//    dfts.resize(nBands);
//    X.resize(nSamps);
//    Y.resize(nSamps);
    
//    VectorXd logBax = bax.log2();
//    VectorXd logFax = fax.log2();
//    
//    double c = log(16) / (square(1.0 / ppo));
//    MatrixXd logDiff = logFax.rowwise() - logFax.transpose();
//    MatrixXd g_ = (-c * logDiff.square()).exp();
//    
//    //    int end = nBands - 1;
//    //    g.col(0)   = (g.col( 0 ) < bax( 0 )).select(1, bax( 0 ));
//    //    g.col(end) = (g.col(end) < bax(end)).select(1, bax(end));
////    g_.for_each([](double& val){ if (val < th) val = 0; });
//    
//    d = g_.square().rowwise().sum();
//    MatrixXd gDual_ = g_.colwise() / d;
//    
//    using namespace std::complex_literals;
//    
////    for (size_t k = 0; k < nBands; k++) {
////        VectorXi ii = g_.col(k) != 0;
////        ii = padIdxs(ii);
////        assert(ii.size() >= 4);
////        
////        uword offset = ii(0);
////        uword nCoefs = ii.size();
////        scale(k) = double(nCoefs);
////        phase(k) = exp(1i * datum::tau * double(offset) * regspace<VectorXd>(0, nCoefs-1)/double(nCoefs));
////        idx(k) = span(ii(0), ii(ii.n_elem-1));
////        g(k) = g_(idx(k), k);
////        gDual(k) = gDual_(idx(k), k);
////        dfts(k).init(nCoefs);
////    }
//    
//    Xi = getCoefs();
//    Yi = getCoefs();
//    
//    dft.init(nSamps);
}

void NsgfCqtSparse::forward(const vec& x, CqtCoefs& Xcq) {
    assert(Xcq.size() == nBands);
//    X.fill(0);
//    dft.rdft(x, X);
//    X /= double(nSamps);
//    for (size_t k = 0; k < nBands; k++) {
//        Xi[k] = X(idx[k].first, idx[k].second) * g[k];
//        dfts[k].idft(Xi[k], Xcq[k]);
//        Xcq[k] *= scale(k);
//        Xcq[k] *= phase[k];
//    }
}

void NsgfCqtSparse::inverse(const CqtCoefs& Xcq, vec& x) {
    assert(Xcq.size() == nBands);
//    Y.fill(0);
//    for (size_t k = 0; k < nBands; k++) {
//        Yi[k] = Xcq[k];
//        Yi[k] *= phase[k].conjugate();
//        Yi[k] /= scale(k);
//        dfts[k].dft(Yi[k], Yi[k]);
//        Yi[k] = Yi[k] * g[k];
////        Y(idx[k].first, idx[k].second) = Y(idx[k].first, idx[k].second) + Yi[k];
//    }
//    dft.irdft(Y, x);
//    x *= nSamps;
}

uvec NsgfCqtSparse::padIdxs(uvec ii) {
//    int i0 = ii(0);
//    int nIdx = ii.size();
//    if (nIdx < 4) nIdx = 4;
////    nIdx = nextPow2(nIdx);
//    ii = VectorXi::LinSpaced(i0, i0 + nIdx-1);
    return ii;
}

CqtCoefs NsgfCqtSparse::getCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    CqtCoefs coefs(nBands);
    for (size_t k = 0; k < nBands; k++) {
        coefs[k].resize(g[k].size());
    }
    return coefs;
}
