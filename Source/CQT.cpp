//
//  CQT.cpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#include "CQT.hpp"

#include "RTChecker.h"
#include "MathUtils.h"

#include "../Tests/Source/VectorOps.h"

using namespace jsa;
using namespace Eigen;

NsgfCqtCommon::NsgfCqtCommon(double sampleRate, Index numSamples,
                             double fraction, double minFrequency,
                             double maxFrequency, double refFrequency) :
    fs(sampleRate), nSamps(numSamples), frac(fraction), fMin(minFrequency),
    fMax(maxFrequency), fRef(refFrequency),
    bandInfo(computeBandInfo(frac, fMin, fMax, fRef)),
    nBands(bandInfo.nBands), nFreqs(nSamps), bax(nBands), fax(nFreqs),
    d(nFreqs), Xdft(nSamps), dft(nSamps)
{
    Xdft.setZero();
    bax = fRef * (frac * log(2) * regspace(-bandInfo.nBandsDown, bandInfo.nBandsUp)).exp();
    fax = ArrayXd::LinSpaced(nFreqs, 0, nFreqs - 1) * fs / double(nFreqs);
}

//==========================================================================
//==========================================================================
//==========================================================================

NsgfCqtFull::NsgfCqtFull(double sampleRate, Index numSamples,
                         double fraction, double minFrequency,
                         double maxFrequency, double refFrequency) :
    NsgfCqtCommon(sampleRate, numSamples, fraction, minFrequency, maxFrequency,
                  refFrequency),
    Xmat(nSamps, nBands)
{
    double c = log(4) / (square(frac));
    ArrayXXd outerDif = fax.log2().rowwise().replicate(bax.size()) -
        bax.log2().transpose().colwise().replicate(fax.size());
    g = (-c * outerDif.square()).exp();
    
    Index end = nBands - 1;
    g.col(0) = (fax < bax(0)).select(1, g.col(0));
    g.col(end) = (fax > bax(end)).select(1, g.col(end));
    d = g.square().rowwise().sum();
    gDual = g.colwise() / d;
    
    g.bottomRows(nFreqs / 2 - 1).setZero();
    gDual.bottomRows(nFreqs / 2 - 1).setZero();
    
    Xmat.setZero();
}

void NsgfCqtFull::forward(const ArrayXd &x, ArrayXXcd &Xcq) {
    RealTimeChecker ck;
    
    assert(x.size() == nSamps);
    assert(Xcq.cols() == Index(nBands));
    assert(Xcq.rows() == Index(nSamps));
    dft.rdft(x, Xdft);
    for (Index k = 0; k < nBands; k++) {
        Xmat.col(k) = 2 * g.col(k) * Xdft;
    }
    dft.idft(Xmat, Xcq);
}

void NsgfCqtFull::inverse(const ArrayXXcd &Xcq, ArrayXd &x) {
    RealTimeChecker ck;
    
    assert(x.size() == nSamps);
    assert(Xcq.cols() == Index(nBands));
    assert(Xcq.rows() == Index(nSamps));
    dft.dft(Xcq, Xmat);
    Xdft = (Xmat * gDual).rowwise().sum() / 2;
    dft.irdft(Xdft, x);
}

//==========================================================================
//==========================================================================
//==========================================================================

NsgfCqtSparse::Span findIdx(const ArrayXd &x, double th) {
    Index i0 = 0;
    Index len = 0;
    for (Index i = 0; i < x.size(); i++) {
        if (x(i) > th) {
            i0 = i;
            break;
        }
    }
    
    for (Index i = i0; i < x.size(); i++) {
        len++;
        if (x(i) < th) {
            break;
        }
    }
    return {i0, len};
}

NsgfCqtSparse::NsgfCqtSparse(double sampleRate, Index numSamples,
                             double fraction, double minFrequency,
                             double maxFrequency, double refFrequency) :
    NsgfCqtCommon(sampleRate, numSamples, fraction, minFrequency, maxFrequency,
                  refFrequency),
    idx(nBands), g(nBands), gDual(nBands), phase(nBands), scale(nBands),
    dfts(nBands)
{
    double c = log(4) / (square(frac));
    ArrayXXd outerDif = fax.log2().rowwise().replicate(bax.size()) -
        bax.log2().transpose().colwise().replicate(fax.size());
    ArrayXXd g_ = (-c * outerDif.square()).exp();
    
    Index end = nBands - 1;
    g_.col(0) = (fax < bax(0)).select(1, g_.col(0));
    g_.col(end) = (fax > bax(end)).select(1, g_.col(end));
    g_ = (g_ <= th).select(0.0, g_);
    
    d = g_.square().rowwise().sum();
    ArrayXXd gDual_ = g_.colwise() / d;
    
    g_.bottomRows(nFreqs / 2 - 1).fill(0);
    gDual_.bottomRows(nFreqs / 2 - 1).fill(0);
    
    using namespace std::complex_literals;
    
    for (Index k = 0; k < nBands; k++) {
        idx[k] = getIdx(g_.col(k));
        Index i0 = idx[k].i0;
        Index len = idx[k].len;
        scale(k) = double(len);
        ArrayXd n = regspace(len);
        phase[k] = exp(1i * 2.0 * M_PI * double(i0) * n / double(len));
        g[k] = g_.col(k).segment(i0, len);
        gDual[k] = gDual_.col(k).segment(i0, len);
        dfts[k].reset(new DFT(len));
    }
    
    Xcoefs = getCoefs();
}

void NsgfCqtSparse::forward(const ArrayXd &x, Coefs &Xcq) {
    RealTimeChecker ck;
    
    if (fs < 0) return;
    assert(Index(Xcq.size()) == nBands);
    Xdft.fill(0);
    dft.rdft(x, Xdft);
    Xdft /= nSamps;
    for (Index k = 0; k < nBands; k++) {
        Xcoefs[k] = g[k] * Xdft.segment(idx[k].i0, idx[k].len);
        dfts[k]->idft(Xcoefs[k], Xcoefs[k]);
        Xcq[k] = 2.0 * idx[k].len * phase[k] * Xcoefs[k];
    }
}

void NsgfCqtSparse::inverse(const Coefs &Xcq, ArrayXd &x) {
    RealTimeChecker ck;
    
    if (fs < 0) return;
    assert(Index(Xcq.size()) == nBands);
    Xdft.fill(0);
    for (Index k = 0; k < nBands; k++) {
        Xcoefs[k] = 1.0 / (2.0 * idx[k].len) * phase[k].conjugate() * Xcq[k];
        dfts[k]->dft(Xcoefs[k], Xcoefs[k]);
        Xdft.segment(idx[k].i0, idx[k].len) += gDual[k] * Xcoefs[k];
    }
    dft.irdft(Xdft, x);
    x *= nSamps;
}

NsgfCqtSparse::Span NsgfCqtSparse::getIdx(const ArrayXd &x) {
    Index i0 = 0;
    Index i1 = x.size();
    for (Index i = 0; i < x.size(); i++) {
        if (x(i) < th) continue;
        i0 = i; break;
    }
    
    for (Index i = x.size() - 1; i >= 0; i--) {
        if (x(i) < th) continue;
        i1 = i; break;
    }
    
    Index len = i1 - i0 + 1;
    if (len < 4) len = 4;
    len = Index(nextPow2((unsigned int)(len)));
    return {i0, len};
}

NsgfCqtSparse::Frame NsgfCqtSparse::getRealCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Frame frame(nBands);
    for (Index k = 0; k < nBands; k++) {
        Index sz = g[k].size();
        frame[k].resize(sz);
    }
    return frame;
}

NsgfCqtSparse::Coefs NsgfCqtSparse::getCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Coefs coefs(nBands);
    for (Index k = 0; k < nBands; k++) {
        Index sz = g[k].size();
        coefs[k].resize(sz);
        coefs[k].setZero();
    }
    return coefs;
}

NsgfCqtSparse::Coefs NsgfCqtSparse::getValidCoefs() const {
    assert(fs > 0);
    assert(bax.size() > 0);
    Coefs coefs(nBands);
    for (Index k = 0; k < nBands; k++) {
        Index sz = g[k].size();
        assert(sz % 2 == 0);
        coefs[k].resize(sz / 2);
        coefs[k].setZero();
    }
    return coefs;
}
