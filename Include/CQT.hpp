//
//  CQT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include <Eigen/Core>

#include "FFT.hpp"

namespace jsa {

/**
 * @class NsgfCqtCommon
 * @brief Base class for Non-Stationary Gabor Filterbank Constant-Q Transform (NSGF-CQT) operations.
 * 
 * This class provides common functionality and data members for NSGF-CQT operations, 
 * including frequency axis computation, band information, and diagonalization.
 */
class NsgfCqtCommon {
public:
    /**
     * @struct BandInfo
     * @brief Holds information about the number of bands in the filterbank.
     */
    struct BandInfo {
        Eigen::Index nBands;     ///< Total number of bands.
        Eigen::Index nBandsDown; ///< Number of bands below the reference frequency.
        Eigen::Index nBandsUp;   ///< Number of bands above the reference frequency.
    };
    
    /**
     * @brief Constructor for NsgfCqtCommon.
     *
     * @param sampleRate Sampling rate of the signal (Hz).
     * @param numSamples Number of samples in a block.
     * @param fraction Fractional bandwidth of the filterbank.
     * @param minFrequency Minimum frequency of the filterbank (Hz).
     * @param maxFrequency Maximum frequency of the filterbank (Hz).
     * @param refFrequency Reference frequency for the filterbank (Hz).
     */
    NsgfCqtCommon(double sampleRate, Eigen::Index numSamples, double fraction,
                  double minFrequency, double maxFrequency, double refFrequency);
    
    // Accessor methods for various parameters and computed data.
    double getSampleRate() const { return fs; }
    Eigen::Index getBlockSize() const { return nSamps; }
    Eigen::Index getNumSamps() const { return nSamps; }
    double getFraction() const { return frac; }
    double getPpo() const { return 1.0/frac; }
    double getMinFreq() const { return fMin; }
    double getMaxFreq() const { return fMax; }
    double getRefFreq() const { return fRef; }
    Eigen::Index getNumFreqs() const { return nFreqs; }
    Eigen::Index getNumBands() const { return nBands; }
    const Eigen::ArrayXd& getFrequencyAxis() const { return fax; }
    const Eigen::ArrayXd& getBandAxis() const { return bax; }
    const Eigen::ArrayXd& getDiagonalization() const { return d; }

protected:
    /**
     * @brief Computes band information for the filterbank.
     * 
     * @param frac Fractional bandwidth of the filterbank.
     * @param fMin Minimum frequency of the filterbank (Hz).
     * @param fMax Maximum frequency of the filterbank (Hz).
     * @param fRef Reference frequency for the filterbank (Hz).
     * @return BandInfo Struct containing band information.
     */
    inline constexpr BandInfo computeBandInfo(double frac, double fMin,
                                              double fMax, double fRef)
    {
        Eigen::Index nBandsUp = Eigen::Index(ceil(1.0/frac * log2(fMax / fRef)));
        Eigen::Index nBandsDown = Eigen::Index(ceil(1.0/frac * log2(fRef / fMin)));
        Eigen::Index nBands = nBandsDown + nBandsUp + 1;
        return { nBands, nBandsDown, nBandsUp };
    }

    // Member variables for filterbank parameters and computed data.
    const double fs;              ///< Sampling rate (Hz).
    const Eigen::Index nSamps;    ///< Number of samples in a block.
    const double frac;            ///< Fractional bandwidth.
    const double fMin;            ///< Minimum frequency (Hz).
    const double fMax;            ///< Maximum frequency (Hz).
    const double fRef;            ///< Reference frequency (Hz).
    const BandInfo bandInfo;      ///< Band information.
    const Eigen::Index nBands;    ///< Total number of bands.
    const Eigen::Index nFreqs;    ///< Number of frequencies.
    Eigen::ArrayXd bax;           ///< Band axis.
    Eigen::ArrayXd fax;           ///< Frequency axis.
    Eigen::ArrayXd d;             ///< Diagonalization array.
    Eigen::ArrayXcd Xdft;         ///< DFT of the signal.
    DFT dft;                      ///< Discrete Fourier Transform object.
};

/**
 * @class NsgfCqtFull
 * @brief Full implementation of the NSGF-CQT.
 * 
 * This class provides methods for forward and inverse NSGF-CQT transformations
 * using a full representation of the filterbank.
 */
class NsgfCqtFull : public NsgfCqtCommon {
public:
    /**
     * @brief Constructor for NsgfCqtFull.
     * 
     * @param sampleRate Sampling rate of the signal (Hz).
     * @param numSamples Number of samples in the signal.
     * @param fraction Fractional bandwidth of the filterbank.
     * @param minFrequency Minimum frequency of the filterbank (Hz).
     * @param maxFrequency Maximum frequency of the filterbank (Hz).
     * @param refFrequency Reference frequency for the filterbank (Hz).
     */
    NsgfCqtFull(double sampleRate, Eigen::Index numSamples, double fraction,
                double minFrequency, double maxFrequency, double refFrequency);

    /**
     * @brief Performs the forward NSGF-CQT transformation.
     * 
     * @param x Input signal.
     * @param Xcq Output constant-Q transform coefficients.
     */
    void forward(const Eigen::ArrayXd& x, Eigen::ArrayXXcd& Xcq);

    /**
     * @brief Performs the inverse NSGF-CQT transformation.
     * 
     * @param Xcq Input constant-Q transform coefficients.
     * @param x Output reconstructed signal.
     */
    void inverse(const Eigen::ArrayXXcd& Xcq, Eigen::ArrayXd& x);

    // Accessor methods for frame and dual frame.
    const Eigen::ArrayXXd& getFrame() const { return g; }
    const Eigen::ArrayXXd& getDualFrame() const { return gDual; }

private:
    Eigen::ArrayXXd g;       ///< Frame matrix.
    Eigen::ArrayXXd gDual;   ///< Dual frame matrix.
    Eigen::ArrayXXcd Xmat;   ///< Matrix of transform coefficients.
};

/**
 * @class NsgfCqtSparse
 * @brief Sparse implementation of the NSGF-CQT.
 * 
 * This class provides methods for forward and inverse NSGF-CQT transformations
 * using a sparse representation of the filterbank.
 */
class NsgfCqtSparse : public NsgfCqtCommon {
public:
    /**
     * @struct Span
     * @brief Represents a span of indices for a band.
     */
    struct Span {
        Eigen::Index i0 = 0; ///< Starting index of the span.
        Eigen::Index len = 0; ///< Length of the span.
    };

    using Coefs = std::vector<Eigen::ArrayXcd>; ///< Type alias for coefficients.
    using Frame = std::vector<Eigen::ArrayXd>;  ///< Type alias for frame.
    using SpanList = std::vector<Span>;         ///< Type alias for span list.

    /**
     * @brief Constructor for NsgfCqtSparse.
     * 
     * @param sampleRate Sampling rate of the signal (Hz).
     * @param nSamps Number of samples in the signal.
     * @param fraction Fractional bandwidth of the filterbank.
     * @param minFrequency Minimum frequency of the filterbank (Hz).
     * @param maxFrequency Maximum frequency of the filterbank (Hz).
     * @param refFrequency Reference frequency for the filterbank (Hz).
     */
    NsgfCqtSparse(double sampleRate, Eigen::Index nSamps, double fraction,
                  double minFrequency, double maxFrequency, double refFrequency);

    /**
     * @brief Performs the forward NSGF-CQT transformation.
     * 
     * @param x Input signal.
     * @param Xcq Output constant-Q transform coefficients.
     */
    void forward(const Eigen::ArrayXd& x, Coefs& Xcq);

    /**
     * @brief Performs the inverse NSGF-CQT transformation.
     * 
     * @param Xcq Input constant-Q transform coefficients.
     * @param x Output reconstructed signal.
     */
    void inverse(const Coefs& Xcq, Eigen::ArrayXd& x);

    // Accessor methods for frame, dual frame, and band spans.
    const Frame& getFrame() const { return g; }
    const Eigen::ArrayXd& getAtom(Eigen::Index k) const { return g[k]; }
    const Frame& getDualFrame() const { return gDual; }
    const Eigen::ArrayXd& getDualAtom(Eigen::Index k) const { return gDual[k]; }
    const Span getBandSpan(Eigen::Index k) const { return idx[k]; }
    Eigen::ArrayXd getFrequencyAxis(Eigen::Index k) const { return fax.segment(idx[k].i0, idx[k].len); }
    Eigen::Index getLength(Eigen::Index k) const { return idx[k].len; };
    const Coefs& getPhaseCoefs() const { return phase; }

    // Methods for retrieving coefficients.
    Frame getRealCoefs() const;
    Coefs getCoefs() const;
    Coefs getValidCoefs() const;

private:
    /**
     * @brief Computes the span of indices for a given array.
     * 
     * @param ii Input array of indices.
     * @return Span Struct representing the span of indices.
     */
    Span getIdx(const Eigen::ArrayXd& ii);

    SpanList idx;                           ///< List of spans for each band.
    Frame g;                                ///< Frame representation.
    Frame gDual;                            ///< Dual frame representation.
    Coefs phase;                            ///< Phase coefficients.
    Eigen::ArrayXd scale;                   ///< Scale factors.
    Coefs Xcoefs;                           ///< Sparse coefficients.
    std::vector<std::unique_ptr<DFT>> dfts; ///< DFT objects for each band.
    inline static const double th = 1e-6;   ///< Threshold for sparsity.
};

}
