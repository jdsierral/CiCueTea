//
//  CQT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

/**
 * @file CQT.hpp
 * @brief Provides an implementation of CQT and its inverses.
 * @author Juan Sierra
 * @date 3/9/25
 * @copyright MIT License
 */

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "FFT.hpp"

namespace jsa {

/**
 * @class NsgfCqtCommon
 * @brief Base class for Non-Stationary Gabor Filterbank Constant-Q Transform (NSGF-CQT) operations.
 * 
 * This class provides common functionality and data members for NSGF-CQT operations, 
 * including frequency axis computation, band information, and diagonalization.
 *
 * Construction never throws and never crashes: a configuration that fails
 * validate() produces an *inert* object — isValid() returns false, the
 * processing methods output silence, and accessors return empty data. This
 * supports host lifecycles (e.g. DAWs) that construct with a placeholder
 * sample rate before the real one is known; since the whole design depends
 * on the sample rate, the correct response to a new rate is constructing a
 * new object.
 */
class NsgfCqtCommon
{
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
     * @param fraction Reciprocal of bands per octave (e.g. 1.0/12 for 12 bands/octave); fractional values allowed.
     * @param minFrequency Minimum frequency of the filterbank (Hz).
     * @param maxFrequency Maximum frequency of the filterbank (Hz).
     * @param refFrequency Reference frequency for the filterbank (Hz).
     */
    NsgfCqtCommon(double sampleRate, Eigen::Index numSamples, double fraction,
                  double minFrequency, double maxFrequency, double refFrequency);

    /**
     * @brief True when the object is usable: the configuration passed
     * validate() *and* the constructed frame passed the measured health
     * check (see checkFrameHealth()).
     *
     * Two tiers on purpose: parameter rules catch the provable absurdities,
     * but the fuzzy boundary — a convergence of factors with no closed-form
     * criterion — is decided by inspecting the frame that was actually
     * built, not by predicting from parameters. Two measured modes:
     * coverage gaps (bins no atom reaches: frame-operator condition number,
     * checkFrameHealth()) and unresolved atoms (bands too narrow for the
     * grid: per-atom support ≥ minAtomSupport).
     *
     * An invalid object is inert: forward()/inverse() write zeros to their
     * outputs and return, and accessors return empty or unusable data.
     * Parameter getters (getSampleRate() etc.) still report what was passed.
     */
    bool isValid() const { return valid && frameOk; }

    /**
     * @brief Condition number of the painless frame operator, max(d)/min(d)
     * up to Nyquist. Advisory: reconstruction error amplifies by roughly
     * cond·ε, so values ≲ 1e6 keep double-precision round trips near 1e-10.
     * Returns +inf when the object is invalid.
     */
    double getFrameConditionNumber() const;

    // Accessor methods for various parameters and computed data.
    double                getSampleRate() const { return fs; }
    Eigen::Index          getBlockSize() const { return nSamps; } ///< Synonym of getNumSamps().
    Eigen::Index          getNumSamps() const { return nSamps; }  ///< Synonym of getBlockSize().
    double                getFraction() const { return frac; }
    double                getPpo() const { return 1.0 / frac; }
    double                getMinFreq() const { return fMin; }
    double                getMaxFreq() const { return fMax; }
    double                getRefFreq() const { return fRef; }
    Eigen::Index          getNumFreqs() const { return nFreqs; }
    Eigen::Index          getNumBands() const { return nBands; }
    const Eigen::ArrayXd& getFrequencyAxis() const { return fax; }
    const Eigen::ArrayXd& getBandAxis() const { return bax; }
    const Eigen::ArrayXd& getDiagonalization() const { return d; }

  protected:
    /**
     * @brief Checks the structural validity conditions for a configuration.
     *
     * Deliberately limited to conditions under which the transform is
     * provably meaningless: positive sample rate, power-of-two block size
     * (a structural assumption of the per-band span logic, see
     * NsgfCqtSparse::getIdx — independent of what a backend could handle),
     * positive fraction and reference frequency, 0 < fMin < fMax, and fMax
     * strictly below Nyquist. Nothing about the *extent* of the range is
     * gated (sub-octave ranges are legitimate): feasibility is measured on
     * the constructed frame instead (checkFrameHealth(), atom support).
     */
    static bool validate(double fs, Eigen::Index nSamps, double frac,
                         double fMin, double fMax, double fRef);

    /**
     * @brief Computes band information for the filterbank.
     * 
     * @param frac Reciprocal of bands per octave.
     * @param fMin Minimum frequency of the filterbank (Hz).
     * @param fMax Maximum frequency of the filterbank (Hz).
     * @param fRef Reference frequency for the filterbank (Hz).
     * @return BandInfo Struct containing band information.
     */
    static BandInfo computeBandInfo(double frac, double fMin,
                                    double fMax, double fRef);

    /**
     * @brief Measured frame-health check on the constructed frame operator.
     *
     * The painless frame operator is diagonal with entries d(f); the
     * transform is invertible iff min(d) > 0 on the covered range, and
     * numerically trustworthy iff the condition number max(d)/min(d) is
     * moderate. Requiring min(d) > dHealthTol·max(d) bounds the condition
     * number by 1/dHealthTol. Called by the derived constructors once d is
     * available; every "Q too high for this block" failure mode lands here,
     * whatever combination of parameters caused it.
     */
    bool checkFrameHealth() const;

    /// Relative floor for the frame-operator diagonal: bounds the frame
    /// condition number by 1e6, keeping double-precision round trips ~1e-10.
    static constexpr double dHealthTol = 1e-6;

    /// Atom magnitude threshold: below it a bin does not count as support
    /// (and, in the sparse variant, is zeroed for efficiency).
    static constexpr double th = 1e-6;

    /// Minimum number of bins above `th` an atom must occupy to count as
    /// resolved by the frequency grid. The fuzzy "Q too big for the block"
    /// failure mode is invisible in d — with many bands per octave every bin
    /// is still covered — so it is caught from the atom side instead:
    /// unresolved atoms are the measured, warp-agnostic equivalent of the
    /// parametric Q check. 4 matches the smallest FFT span getIdx will build
    /// (and min_bw in the Python reference).
    static constexpr Eigen::Index minAtomSupport = 4;

    // Member variables for filterbank parameters and computed data.
    // `valid` is declared first on purpose: member initialization follows
    // declaration order, and the members below branch on it.
    const bool         valid;    ///< Structural validity: result of validate().
    bool               frameOk = true; ///< Measured frame health; set by derived ctors.
    const double       fs;       ///< Sampling rate (Hz).
    const Eigen::Index nSamps;   ///< Number of samples in a block (0 when invalid).
    const double       frac;     ///< Reciprocal of bands per octave.
    const double       fMin;     ///< Minimum frequency (Hz).
    const double       fMax;     ///< Maximum frequency (Hz).
    const double       fRef;     ///< Reference frequency (Hz).
    const BandInfo     bandInfo; ///< Band information.
    const Eigen::Index nBands;   ///< Total number of bands.
    const Eigen::Index nFreqs;   ///< Number of frequencies.
    Eigen::ArrayXd     bax;      ///< Band axis.
    Eigen::ArrayXd     fax;      ///< Frequency axis.
    Eigen::ArrayXd     d;        ///< Diagonalization array.
    Eigen::ArrayXcd    Xdft;     ///< DFT of the signal.
    DFT                dft;      ///< Discrete Fourier Transform object.
};

/**
 * @class NsgfCqtDense
 * @brief Dense implementation of the NSGF-CQT.
 * 
 * This class provides methods for forward and inverse NSGF-CQT transformations
 * using a dense representation of the filterbank.
 */
class NsgfCqtDense : public NsgfCqtCommon
{
  public:
    /**
     * @brief Constructor for NsgfCqtDense.
     * 
     * @param sampleRate Sampling rate of the signal (Hz).
     * @param numSamples Number of samples in the signal.
     * @param fraction Reciprocal of bands per octave (e.g. 1.0/12 for 12 bands/octave); fractional values allowed.
     * @param minFrequency Minimum frequency of the filterbank (Hz).
     * @param maxFrequency Maximum frequency of the filterbank (Hz).
     * @param refFrequency Reference frequency for the filterbank (Hz).
     */
    NsgfCqtDense(double sampleRate, Eigen::Index numSamples, double fraction,
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
    Eigen::ArrayXXd  g;     ///< Frame matrix.
    Eigen::ArrayXXd  gDual; ///< Dual frame matrix.
    Eigen::ArrayXXcd Xmat;  ///< Matrix of transform coefficients.
};

/**
 * @class NsgfCqtSparse
 * @brief Sparse implementation of the NSGF-CQT.
 * 
 * This class provides methods for forward and inverse NSGF-CQT transformations
 * using a sparse representation of the filterbank.
 */
class NsgfCqtSparse : public NsgfCqtCommon
{
  public:
    /**
     * @struct Span
     * @brief Represents a span of indices for a band.
     */
    struct Span {
        Eigen::Index i0  = 0; ///< Starting index of the span.
        Eigen::Index len = 0; ///< Length of the span.
    };

    using Coefs    = std::vector<Eigen::ArrayXcd>; ///< Type alias for coefficients.
    using Frame    = std::vector<Eigen::ArrayXd>;  ///< Type alias for frame.
    using SpanList = std::vector<Span>;            ///< Type alias for span list.

    /**
     * @brief Constructor for NsgfCqtSparse.
     * 
     * @param sampleRate Sampling rate of the signal (Hz).
     * @param nSamps Number of samples in the signal.
     * @param fraction Reciprocal of bands per octave (e.g. 1.0/12 for 12 bands/octave); fractional values allowed.
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
    const Frame&          getFrame() const { return g; }
    const Eigen::ArrayXd& getAtom(Eigen::Index k) const { return g[k]; }
    const Frame&          getDualFrame() const { return gDual; }
    const Eigen::ArrayXd& getDualAtom(Eigen::Index k) const { return gDual[k]; }
    const Coefs&          getPhaseCoefs() const { return phase; }
    Span                  getBandSpan(Eigen::Index k) const { return idx[k]; }
    Eigen::ArrayXd        getFrequencyAxis(Eigen::Index k) const { return fax.segment(idx[k].i0, idx[k].len); }
    Eigen::Index          getLength(Eigen::Index k) const { return idx[k].len; };
    double                getCoeffRate(Eigen::Index k) const { return getSampleRate() * double(getLength(k)) / double(getBlockSize()); }

    // Methods for retrieving coefficients.
    Frame getRealCoefs() const;
    Coefs getCoefs() const;
    Coefs getValidCoefs() const;

  private:
    /**
     * @brief Derives the index span a band occupies on the frequency grid.
     *
     * Scans for the first and last indices whose value exceeds the sparsity
     * threshold, then enforces a "correct" span: at least 4 bins long and
     * rounded up to a power of two, so each band gets an efficient FFT size.
     *
     * @param ii The band's profile over frequency-grid indices.
     * @return Span Power-of-two-length index span covering the band's support.
     */
    Span getIdx(const Eigen::ArrayXd& ii);

    SpanList                          idx;       ///< List of spans for each band.
    Frame                             g;         ///< Frame representation.
    Frame                             gDual;     ///< Dual frame representation.
    Coefs                             phase;     ///< Phase coefficients.
    Eigen::ArrayXd                    scale;     ///< Per-band scale (= span length). Stored as double
    Coefs                             Xcoefs;    ///< Sparse coefficients.
    std::vector<std::unique_ptr<DFT>> dfts;      ///< DFT objects for each band.
};

} // namespace jsa
