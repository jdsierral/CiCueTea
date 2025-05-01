//
//  CQT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include "FFT.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace jsa {

class NsgfCqtCommon {
public:
    struct BandInfo {
        Eigen::Index nBands;
        Eigen::Index nBandsDown;
        Eigen::Index nBandsUp;
    };
    
    NsgfCqtCommon(double sampleRate, Eigen::Index blockSize, double fraction,
                  double minFrequency, double maxFrequency, double refFrequency);
    
    double getSampleRate() const { return fs; }
    Eigen::Index getBlockSize() const { return nSamps; }
    Eigen::Index getNumSamps() const { return nSamps; }
    double getFraction() const { return frac; }
    double getPpo() const { return 1.0 / frac; }
    double getMinFreq() const { return fMin; }
    double getMaxFreq() const { return fMax; }
    double getRefFreq() const { return fRef; }
    Eigen::Index getNumFreqs() const { return nFreqs; }
    Eigen::Index getNumBands() const { return nBands; }
    
    const Eigen::ArrayXd& getFrequencyAxis() const { return fax; }
    const Eigen::ArrayXd& getBandAxis() const { return bax; }
    const Eigen::ArrayXd& getDiagonalization() const { return d; }

protected:
    
    inline constexpr BandInfo computeBandInfo(double frac, double fMin,
                                              double fMax, double fRef)
    {
        Eigen::Index nBandsUp = Eigen::Index(ceil(1.0/frac * log2(fMax / fRef)));
        Eigen::Index nBandsDown = Eigen::Index(ceil(1.0/frac * log2(fRef / fMin)));
        Eigen::Index nBands = nBandsDown + nBandsUp + 1;
        return { nBands, nBandsDown, nBandsUp };
    }

    const double fs;
    const Eigen::Index nSamps;
    const double frac;
    const double fMin;
    const double fMax;
    const double fRef;
    const BandInfo bandInfo;
    const Eigen::Index nBands;
    const Eigen::Index nFreqs;
    
    Eigen::ArrayXd bax;
    Eigen::ArrayXd fax;
    Eigen::ArrayXd d;
    Eigen::ArrayXcd Xdft;
        
    DFT dft;
};

class NsgfCqtFull : public NsgfCqtCommon {
public:
    NsgfCqtFull(double sampleRate, Eigen::Index numSamples, double fraction,
                double minFrequency, double maxFrequency, double refFrequency);
    void forward(const  Eigen::ArrayXd &  x , Eigen::ArrayXXcd& Xcq);
    void inverse(const Eigen::ArrayXXcd& Xcq, Eigen::ArrayXd&    x );
    
    const Eigen::ArrayXXd& getFrame() const { return g; }
    const Eigen::ArrayXXd& getDualFrame() const { return gDual; }
    
private:
    Eigen::ArrayXXd g;
    Eigen::ArrayXXd gDual;
    Eigen::ArrayXXcd Xmat;
};

class NsgfCqtSparse : public NsgfCqtCommon {
public:
    struct Span { Eigen::Index i0 = 0; Eigen::Index len = 0; };
    using Coefs = std::vector<Eigen::ArrayXcd>;
    using Frame = std::vector<Eigen::ArrayXd>;
    using SpanList = std::vector<Span>;
    
    NsgfCqtSparse(double sampleRate, Eigen::Index nSamps, double fraction,
                  double minFrequency, double maxFrequency, double refFrequency);
    
    void forward(const Eigen::ArrayXd& x, Coefs& Xcq);
    void inverse(const Coefs& Xcq, Eigen::ArrayXd& x );
    
    const Frame& getFrame() const { return g; }
    const Eigen::ArrayXd& getAtom(Eigen::Index k) const { return g[k]; }
    const Frame& getDualFrame() const { return gDual; }
    const Eigen::ArrayXd& getDualAtom(Eigen::Index k) const { return gDual[k]; }
    const Span getBandSpan(Eigen::Index k) const { return idx[k]; }
    Eigen::ArrayXd getFrequencyAxis(Eigen::Index k) const {
        auto ii = idx[k];
        return fax.segment(ii.i0, ii.len);
    }

    Frame getRealCoefs() const;
    Coefs getCoefs() const;
    Coefs getValidCoefs() const;

private:
    Span getIdx(const Eigen::ArrayXd& ii);
    
    SpanList idx;
    Frame g;
    Frame gDual;
    Coefs phase;
    Eigen::ArrayXd scale;

    Coefs Xcoefs;
    std::vector<std::unique_ptr<DFT>> dfts;
    inline static const double th = 1e-6;
};

}
