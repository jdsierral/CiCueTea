//
//  Splicer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

/**
 * @file Splicer.hpp
 * @brief Provides an implementation of a Splicer of sliced data to produce 
 * continuous data
 * @author Juan Sierra
 * @date 3/23/25
 * @copyright MIT License
 */

#pragma once

#include <Eigen/Core>

namespace jsa {

/**
 * @class Splicer
 * @brief A class for managing and processing audio blocks with overlap and hop sizes.
 *
 * The `Splicer` class provides functionality to set block sizes, push audio blocks into a buffer,
 * and retrieve individual samples while maintaining overlap and hop size constraints.
 */
class Splicer
{
  public:
    /**
     * @brief Constructor sets the block size and hop size for the splicer.
     *
     * @param newBlockSize The size of the audio block.
     * @param newHopSize The hop size (step size) between consecutive blocks.
     */
    Splicer(Eigen::Index newBlockSize, Eigen::Index newHopSize);

    /**
     * @brief Pushes a new audio block into the buffer.
     *
     * @param block An Eigen array representing the audio block to be added.
     */
    void pushBlock(const Eigen::ArrayXd& block);

    /**
     * @brief Retrieves the next sample from the buffer.
     *
     * @return A double representing the next audio sample.
     */
    double getSample();

    /**
     * @brief Gets the current block size.
     *
     * @return The size of the audio block.
     */
    Eigen::Index getBlockSize() const { return blockSize; }

    /**
     * @brief Gets the current overlap size.
     *
     * @return The size of the overlap between consecutive blocks.
     */
    Eigen::Index getOverlapSize() const { return overlapSize; }

    /**
     * @brief Gets the current hop size.
     *
     * @return The hop size (step size) between consecutive blocks.
     */
    Eigen::Index getHopSize() const { return hopSize; }

    /**
     * @brief Gets the current buffer size.
     *
     * @return The size of the internal buffer.
     */
    Eigen::Index getBufferSize() const { return bufferSize; }

  private:
    Eigen::ArrayXd buffer;      ///< The internal buffer for storing audio data.
    Eigen::Index   bufferSize;  ///< The size of the internal buffer.
    Eigen::Index   blockSize;   ///< The size of the audio block.
    Eigen::Index   overlapSize; ///< The size of the overlap between consecutive blocks.
    Eigen::Index   hopSize;     ///< The hop size (step size) between consecutive blocks.
    size_t         wp = 0;      ///< The write pointer for the buffer.
    size_t         rp = 0;      ///< The read pointer for the buffer.
};

} // namespace jsa
