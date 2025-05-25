//
//  Slicer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#pragma once

#include <Eigen/Core>

/**
 * @namespace jsa
 * @brief Namespace containing the Slicer class and related functionality.
 */

namespace jsa {

/**
 * @class Slicer
 * @brief A class for slicing audio or signal data into overlapping blocks.
 * 
 * The Slicer class provides functionality to process a continuous stream of 
 * samples and divide it into fixed-size blocks with a specified overlap. 
 * This is useful for applications such as audio processing, feature extraction, 
 * and time-domain analysis.
 */
class Slicer {
public:
    /**
     * @brief Sets the block size and hop size for slicing.
     * 
     * @param newBlockSize The size of each block in samples.
     * @param newHopSize The hop size (step size) between consecutive blocks in samples.
     */
    void setSize(Eigen::Index newBlockSize, Eigen::Index newHopSize);

    /**
     * @brief Pushes a single sample into the internal buffer.
     * 
     * @param sample The sample value to be added to the buffer.
     */
    void pushSample(double sample);

    /**
     * @brief Checks if a complete block is available for retrieval.
     * 
     * @return True if a complete block is available, false otherwise.
     */
    bool hasBlock();

    /**
     * @brief Retrieves the next available block as a read-only Eigen::ArrayXd.
     * 
     * @return An Eigen::Map<const Eigen::ArrayXd> representing the block.
     * @note This function assumes that a block is available. Ensure `hasBlock()` 
     *       returns true before calling this function.
     */
    Eigen::Map<const Eigen::ArrayXd> getBlock();

    /**
     * @brief Gets the current block size.
     * 
     * @return The block size in samples.
     */
    Eigen::Index getBlockSize() const;

    /**
     * @brief Gets the current overlap size.
     * 
     * @return The overlap size in samples.
     */
    Eigen::Index getOverlapSize() const;

    /**
     * @brief Gets the current hop size.
     * 
     * @return The hop size in samples.
     */
    Eigen::Index getHopSize() const;

    /**
     * @brief Gets the size of the internal buffer.
     * 
     * @return The buffer size in samples.
     */
    Eigen::Index getBufferSize() const;

private:
    Eigen::ArrayXd buffer; ///< Internal buffer for storing samples.
    Eigen::Index bufferSize; ///< Size of the internal buffer in samples.
    Eigen::Index blockSize; ///< Size of each block in samples.
    Eigen::Index overlapSize; ///< Overlap size between consecutive blocks in samples.
    Eigen::Index hopSize; ///< Hop size (step size) between consecutive blocks in samples.
    size_t wp = 0; ///< Write pointer for the buffer.
    size_t rp = 0; ///< Read pointer for the buffer.
};

}
