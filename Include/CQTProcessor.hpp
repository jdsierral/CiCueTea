//
//  OverlapAddProcessor.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <Eigen/Core>

#include "Splicer.hpp"
#include "Slicer.hpp"
#include "CQT.hpp"
#include "DoubleBuffer.h"

namespace jsa {

/**
 * @class CqtFullProcessor
 * @brief Processes audio samples using a full non-stationary Gabor transform-based CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a full CQT (Constant-Q Transform) implementation.
 */
class CqtFullProcessor {
public:
    /**
     * @brief Constructs a CqtFullProcessor object.
     * 
     * @param sampleRate The sampling rate of the audio signal.
     * @param numSamples The number of samples to process.
     * @param fraction The fraction of overlap between windows.
     * @param minFrequency The minimum frequency of the CQT.
     * @param maxFrequency The maximum frequency of the CQT.
     * @param refFrequency The reference frequency for the CQT.
     */
    CqtFullProcessor(double sampleRate, double numSamples, double fraction,
                     double minFrequency, double maxFrequency, double refFrequency);
    
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
     * @brief Gets the CQT object.
     * 
     * @return A constant reference to the CQT object.
     */
    const NsgfCqtFull& getCqt() const { return cqt; }

protected:
    NsgfCqtFull cqt; ///< The CQT object used for processing.

private:
    Eigen::ArrayXd xi; ///< Internal processing variable.
    Eigen::ArrayXd win; ///< Windowing function.
    Eigen::ArrayXXcd Xcq; ///< CQT coefficients.
    Slicer slicer; ///< Slicer for data segmentation.
    Splicer splicer; ///< Splicer for data reconstruction.
    double fs = -1; ///< Sampling rate.
};

//==========================================================================

/**
 * @class SlidingCQTFullProcessor
 * @brief Processes audio samples using a sliding window full CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a sliding window implementation of the full CQT.
 */
class SlidingCQTFullProcessor {
public:
    /**
     * @brief Constructs a SlidingCQTFullProcessor object.
     * 
     * @param sampleRate The sampling rate of the audio signal.
     * @param numSamples The number of samples to process.
     * @param fraction The fraction of overlap between windows.
     * @param minFrequency The minimum frequency of the CQT.
     * @param maxFrequency The maximum frequency of the CQT.
     * @param refFrequency The reference frequency for the CQT.
     */
    SlidingCQTFullProcessor(double sampleRate, double numSamples, double fraction,
                            double minFrequency, double maxFrequency, double refFrequency);

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
    const NsgfCqtFull& getCqt() const { return cqt; }

protected:
    NsgfCqtFull cqt; ///< The CQT object used for processing.

private:
    Eigen::ArrayXd xi; ///< Internal processing variable.
    DoubleBuffer<Eigen::ArrayXXcd> Xcq; ///< Double buffer for CQT coefficients.
    DoubleBuffer<Eigen::ArrayXXcd> Zcq; ///< Double buffer for intermediate coefficients.
    Eigen::ArrayXXcd Ycq; ///< Intermediate CQT coefficients.
    Eigen::ArrayXd win; ///< Windowing function.
    Slicer slicer; ///< Slicer for data segmentation.
    Splicer splicer; ///< Splicer for data reconstruction.
    double fs = -1; ///< Sampling rate.
};

//==========================================================================

/**
 * @class CqtSparseProcessor
 * @brief Processes audio samples using a sparse non-stationary Gabor transform-based CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a sparse CQT implementation.
 */
class CqtSparseProcessor {
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
    CqtSparseProcessor(double sampleRate, double numSamples, double fraction,
                       double minFrequency, double maxFrequency, double refFrequency);

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
     * @brief Gets the CQT object.
     * 
     * @return A constant reference to the CQT object.
     */
    const NsgfCqtSparse& getCqt() const { return cqt; }

protected:
    NsgfCqtSparse cqt; ///< The CQT object used for processing.

private:
    Eigen::ArrayXd xi; ///< Internal processing variable.
    Eigen::ArrayXd win; ///< Windowing function.
    NsgfCqtSparse::Coefs Xcq; ///< Sparse CQT coefficients.
    Slicer slicer; ///< Slicer for data segmentation.
    Splicer splicer; ///< Splicer for data reconstruction.
    double fs = -1; ///< Sampling rate.
};

//==========================================================================

/**
 * @class SlidingCqtSparseProcessor
 * @brief Processes audio samples using a sliding window sparse CQT.
 * 
 * This class provides methods to process individual samples and blocks of data
 * using a sliding window implementation of the sparse CQT.
 */
class SlidingCqtSparseProcessor {
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
    SlidingCqtSparseProcessor(double sampleRate, double numSamples,
                              double fraction, double minFrequency,
                              double maxFrequency, double refFrequency);

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

protected:
    NsgfCqtSparse cqt; ///< The CQT object used for processing.

private:
    Eigen::ArrayXd xi; ///< Internal processing variable.
    DoubleBuffer<NsgfCqtSparse::Coefs> Xcq; ///< Double buffer for sparse CQT coefficients.
    DoubleBuffer<NsgfCqtSparse::Coefs> Zcq; ///< Double buffer for intermediate coefficients.
    NsgfCqtSparse::Coefs Ycq; ///< Intermediate sparse CQT coefficients.
    Eigen::ArrayXd win; ///< Windowing function.
    NsgfCqtSparse::Frame Win; ///< Frame of CQT windows.
    Slicer slicer; ///< Slicer for data segmentation.
    Splicer splicer; ///< Splicer for data reconstruction.
    double fs = -1; ///< Sampling rate.
};

}
