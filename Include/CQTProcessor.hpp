//
//  OverlapAddProcessor.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

/**
 * @file CQTProcessor.hpp
 * @brief Provides an implementation of Overlap Add Based CQT Processor
 * @author Juan Sierra
 * @date 3/17/25
 * @copyright MIT License
 */

#pragma once

#include <Eigen/Core>

#include "CQT.hpp"
#include "DoubleBuffer.h"
#include "Slicer.hpp"
#include "Splicer.hpp"

namespace jsa {

/**
 * @class CqtDenseProcessor
 * @brief Processes audio samples using a dense non-stationary Gabor transform-based CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a dense CQT (Constant-Q Transform) implementation.
 */
class CqtDenseProcessor
{
  public:
    /**
     * @brief Constructs a CqtDenseProcessor object.
     * 
     * @param sampleRate The sampling rate of the audio signal.
     * @param numSamples The number of samples to process.
     * @param fraction The fraction of overlap between windows.
     * @param minFrequency The minimum frequency of the CQT.
     * @param maxFrequency The maximum frequency of the CQT.
     * @param refFrequency The reference frequency for the CQT.
     */
    CqtDenseProcessor(double sampleRate, Eigen::Index numSamples, double fraction,
                     double minFrequency, double maxFrequency, double refFrequency);

    /**
     * @brief Virtual destructor for safe polymorphic use.
     *
     * Declared virtual to ensure that derived classes can be safely deleted
     * through a base class pointer. This destructor is defaulted as the base class
     * does not require custom cleanup.
     */
    virtual ~CqtDenseProcessor() = default;

    /**
     * @brief Processes a single audio sample.
     * 
     * @param sample The audio sample to process.
     * @return The processed sample.
     */
    double processSample(double sample);

    /**
     * @brief Processes a block of data.
     * 
     * @param block The block of data to process.
     */
    virtual void processBlock(Eigen::ArrayXXcd& block) = 0;

    /**
     * @brief Gets the windowing function.
     *
     * @return A constant reference to the windowing function.
     */
    const Eigen::ArrayXd& getWindow() const { return win; }

    /**
     * @brief Gets the CQT object.
     * 
     * @return A constant reference to the CQT object.
     */
    const NsgfCqtDense& getCqt() const { return cqt; }

    /**
     * @brief Gets the latency produced by the processor.
     *
     * @return Integer number of samples of delay between input and output.
     */
    Eigen::Index getLatency() const { return cqt.getBlockSize(); }

  protected:
    NsgfCqtDense cqt; ///< The CQT object used for processing.

  private:
    Eigen::ArrayXd   xi;      ///< Internal processing variable.
    Eigen::ArrayXd   win;     ///< Windowing function.
    Eigen::ArrayXXcd Xcq;     ///< CQT coefficients.
    Slicer           slicer;  ///< Slicer for data segmentation.
    Splicer          splicer; ///< Splicer for data reconstruction.
    double           fs = -1; ///< Sampling rate.
};

//==========================================================================

/**
 * @class CqtSparseProcessor
 * @brief Processes audio samples using a sparse non-stationary Gabor transform-based CQT.
 *
 * This class provides methods to process individual samples and blocks of data
 * using a sparse CQT implementation.
 */
class CqtSparseProcessor
{
  public:
    /**
     * @brief Constructs a CqtSparseProcessor object.
     *
     * @param sampleRate The sampling rate of the audio signal.
     * @param numSamples The number of samples to process.
     * @param fraction The fraction of overlap between windows.
     * @param minFrequency The minimum frequency of the CQT.
     * @param maxFrequency The maximum frequency of the CQT.
     * @param refFrequency The reference frequency for the CQT.
     */
    CqtSparseProcessor(double sampleRate, Eigen::Index numSamples, double fraction,
                       double minFrequency, double maxFrequency, double refFrequency);

    /**
     * @brief Virtual destructor for safe polymorphic use.
     *
     * Declared virtual to ensure that derived classes can be safely deleted
     * through a base class pointer. This destructor is defaulted as the base class
     * does not require custom cleanup.
     */
    virtual ~CqtSparseProcessor() = default;

    /**
     * @brief Processes a single audio sample.
     *
     * @param sample The audio sample to process.
     * @return The processed sample.
     */
    double processSample(double sample);

    /**
     * @brief Processes a block of data.
     *
     * @param block The block of data to process.
     */
    virtual void processBlock(NsgfCqtSparse::Coefs& block) = 0;

    /**
     * @brief Gets the windowing function.
     *
     * @return A constant reference to the windowing function.
     */
    const Eigen::ArrayXd& getWindow() const { return win; }

    /**
     * @brief Gets the CQT object.
     *
     * @return A constant reference to the CQT object.
     */
    const NsgfCqtSparse& getCqt() const { return cqt; }

    /**
     * @brief Gets the latency produced by the processor.
     *
     * @return Integer number of samples of delay between input and output.
     */
    Eigen::Index getLatency() const { return cqt.getBlockSize(); }

  protected:
    NsgfCqtSparse cqt; ///< The CQT object used for processing.

  private:
    Eigen::ArrayXd       xi;      ///< Internal processing variable.
    Eigen::ArrayXd       win;     ///< Windowing function.
    NsgfCqtSparse::Coefs Xcq;     ///< Sparse CQT coefficients.
    Slicer               slicer;  ///< Slicer for data segmentation.
    Splicer              splicer; ///< Splicer for data reconstruction.
    double               fs = -1; ///< Sampling rate.
};

//==========================================================================

/**
 * @class SlidingCQTDenseProcessor
 * @brief Processes audio samples using a sliding window dense CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a sliding window implementation of the dense CQT.
 */
class SlidingCQTDenseProcessor
{
  public:
    /**
     * @brief Constructs a SlidingCQTDenseProcessor object.
     * 
     * @param sampleRate The sampling rate of the audio signal.
     * @param numSamples The number of samples to process.
     * @param fraction The fraction of overlap between windows.
     * @param minFrequency The minimum frequency of the CQT.
     * @param maxFrequency The maximum frequency of the CQT.
     * @param refFrequency The reference frequency for the CQT.
     */
    SlidingCQTDenseProcessor(double sampleRate, Eigen::Index numSamples, double fraction,
                            double minFrequency, double maxFrequency, double refFrequency);

    /**
     * @brief Virtual destructor for safe polymorphic use.
     *
     * Declared virtual to ensure that derived classes can be safely deleted
     * through a base class pointer. This destructor is defaulted as the base class
     * does not require custom cleanup.
     */
    virtual ~SlidingCQTDenseProcessor() = default;

    /**
     * @brief Processes a single audio sample.
     * 
     * @param sample The audio sample to process.
     * @return The processed sample.
     */
    double processSample(double sample);

    /**
     * @brief Processes a block of data.
     * 
     * @param block The block of data to process.
     */
    virtual void processBlock(Eigen::ArrayXXcd& block) = 0;

    /**
     * @brief Gets the windowing function.
     *
     * @return A constant reference to the windowing function.
     */
    const Eigen::ArrayXd& getWindow() const { return win; }

    /**
     * @brief Gets the CQT object.
     * 
     * @return A constant reference to the CQT object.
     */
    const NsgfCqtDense& getCqt() const { return cqt; }

    /**
     * @brief Gets the latency produced by the processor.
     *
     * @return Integer number of samples of delay between input and output.
     */
    Eigen::Index getLatency() const { return 1.5 * cqt.getBlockSize(); }

  protected:
    NsgfCqtDense cqt; ///< The CQT object used for processing.

  private:
    Eigen::ArrayXd                 xi;      ///< Internal processing variable.
    DoubleBuffer<Eigen::ArrayXXcd> Xcq;     ///< Double buffer for CQT coefficients.
    DoubleBuffer<Eigen::ArrayXXcd> Zcq;     ///< Double buffer for intermediate coefficients.
    Eigen::ArrayXXcd               Ycq;     ///< Intermediate CQT coefficients.
    Eigen::ArrayXd                 win;     ///< Windowing function.
    Slicer                         slicer;  ///< Slicer for data segmentation.
    Splicer                        splicer; ///< Splicer for data reconstruction.
    double                         fs = -1; ///< Sampling rate.
};

//==========================================================================

/**
 * @class SlidingCqtSparseProcessor
 * @brief Processes audio samples using a sliding window sparse CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a sliding window implementation of the sparse CQT.
 */
class SlidingCqtSparseProcessor
{
  public:
    /**
     * @brief Constructs a SlidingCqtSparseProcessor object.
     * 
     * @param sampleRate The sampling rate of the audio signal.
     * @param numSamples The number of samples to process.
     * @param fraction The fraction of overlap between windows.
     * @param minFrequency The minimum frequency of the CQT.
     * @param maxFrequency The maximum frequency of the CQT.
     * @param refFrequency The reference frequency for the CQT.
     */
    SlidingCqtSparseProcessor(double sampleRate, Eigen::Index numSamples,
                              double fraction, double minFrequency,
                              double maxFrequency, double refFrequency);

    /**
     * @brief Virtual destructor for safe polymorphic use.
     *
     * Declared virtual to ensure that derived classes can be safely deleted
     * through a base class pointer. This destructor is defaulted as the base class
     * does not require custom cleanup.
     */
    virtual ~SlidingCqtSparseProcessor() = default;

    /**
     * @brief Processes a single audio sample.
     * 
     * @param sample The audio sample to process.
     * @return The processed sample.
     */
    double processSample(double sample);

    /**
     * @brief Processes a block of data.
     * 
     * @param block The block of data to process.
     */
    virtual void processBlock(NsgfCqtSparse::Coefs& block) = 0;

    /**
     * @brief Gets the windowing function.
     * 
     * @return A constant reference to the windowing function.
     */
    const Eigen::ArrayXd& getWindow() const { return win; }

    /**
     * @brief Gets the CQT window for a specific index.
     * 
     * @param k The index of the CQT window.
     * @return A constant reference to the CQT window.
     */
    const Eigen::ArrayXd& getCqtWindow(Eigen::Index k) const { return Win[k]; }

    /**
     * @brief Gets the CQT object.
     * 
     * @return A constant reference to the CQT object.
     */
    const NsgfCqtSparse& getCqt() const { return cqt; }

    /**
     * @brief Gets the latency produced by the processor.
     *
     * @return Integer number of samples of delay between input and output.
     */
    Eigen::Index getLatency() const { return 1.5 * cqt.getBlockSize(); }

  protected:
    NsgfCqtSparse cqt; ///< The CQT object used for processing.

  private:
    Eigen::ArrayXd                     xi;      ///< Internal processing variable.
    DoubleBuffer<NsgfCqtSparse::Coefs> Xcq;     ///< Double buffer for sparse CQT coefficients.
    DoubleBuffer<NsgfCqtSparse::Coefs> Zcq;     ///< Double buffer for intermediate coefficients.
    NsgfCqtSparse::Coefs               Ycq;     ///< Intermediate sparse CQT coefficients.
    Eigen::ArrayXd                     win;     ///< Windowing function.
    NsgfCqtSparse::Frame               Win;     ///< Frame of CQT windows.
    Slicer                             slicer;  ///< Slicer for data segmentation.
    Splicer                            splicer; ///< Splicer for data reconstruction.
    double                             fs = -1; ///< Sampling rate.
};

} // namespace jsa
