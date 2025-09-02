//
//  Buffers.h
//  CQTDSP
//
//  Created by Juan Sierra on 6/15/25.
//

#pragma once

/**
 * @class MatBuffer
 * @brief A ring buffer for storing and retrieving matrices.
 */
class MatBuffer
{
  public:
    /**
     * @brief Sets the size of the buffer to the next power of 2 and initializes it with a zero matrix.
     * @param newSize The desired size of the buffer.
     * @param zero The zero matrix used for initialization.
     */
    void setSize(Eigen::Index newSize, const Eigen::ArrayXXd& zero)
    {
        newSize = nextPow2(newSize);
        buffer.resize(newSize, zero);
    }

    /**
     * @brief Pushes a matrix into the buffer.
     * @param mat The matrix to push into the buffer.
     */
    void pushMat(const Eigen::ArrayXd& mat)
    {
        wp         = constrain(wp, buffer.size());
        buffer[wp] = mat;
        wp++;
    }

    /**
     * @brief Retrieves a matrix from the buffer.
     * @return A mapped constant reference to the retrieved matrix.
     */
    Eigen::Map<const Eigen::ArrayXXd> getMat()
    {
        rp                                      = constrain(rp, buffer.size());
        auto                              block = buffer[rp];
        Eigen::Map<const Eigen::ArrayXXd> mat(block.data(), block.rows(), block.cols());
        rp++;
        return mat;
    }

  private:
    std::vector<Eigen::ArrayXXd> buffer; ///< The buffer storage for matrices.
    size_t                       rp;     ///< The read pointer.
    size_t                       wp;     ///< The write pointer.
};

/**
 * @class BlockBuffer
 * @brief A ring buffer for storing and retrieving blocks of data.
 */
class BlockBuffer
{
  public:
    /**
     * @brief Sets the size of the buffer to the next power of 2 and initializes it with a zero block.
     * @param newSize The desired size of the buffer.
     * @param zero The zero block used for initialization.
     */
    void setSize(Eigen::Index newSize, const Eigen::ArrayXd& zero)
    {
        newSize = nextPow2(newSize);
        buffer.resize(zero.size(), newSize);
        for (Eigen::Index n = 0; n < buffer.cols(); n++) {
            buffer.col(n) = zero;
        }
        wp = 0;
        rp = 0;
    }

    /**
     * @brief Pushes a block of data into the buffer.
     * @param block The block of data to push into the buffer.
     */
    void pushBlock(const Eigen::ArrayXd& block)
    {
        wp             = constrain(wp, buffer.size());
        buffer.col(wp) = block;
        wp++;
    }

    /**
     * @brief Retrieves a block of data from the buffer.
     * @return A mapped constant reference to the retrieved block.
     */
    Eigen::Map<const Eigen::ArrayXd> getBlock()
    {
        rp                                       = constrain(rp, buffer.size());
        auto                             segment = buffer.col(rp);
        Eigen::Map<const Eigen::ArrayXd> block(segment.data(), segment.size());
        rp++;
        return block;
    }

  private:
    Eigen::ArrayXXd buffer; ///< The buffer storage for blocks of data.
    size_t          rp;     ///< The read pointer.
    size_t          wp;     ///< The write pointer.
};

/**
 * @class RingBuffer
 * @brief A ring buffer for storing and retrieving scalar samples.
 */
class RingBuffer
{
  public:
    /**
     * @brief Sets the size of the buffer to the next power of 2.
     * @param newSize The desired size of the buffer.
     */
    void setSize(Eigen::Index newSize)
    {
        newSize = nextPow2(newSize);
        buffer.resize(newSize);
        wp = 0;
        rp = 0;
    }

    /**
     * @brief Pushes a scalar sample into the buffer.
     * @param sample The scalar sample to push into the buffer.
     */
    void pushSample(double sample)
    {
        wp         = constrain(wp, buffer.size());
        buffer(wp) = sample;
        wp++;
    }

    /**
     * @brief Retrieves a scalar sample from the buffer.
     * @return The retrieved scalar sample.
     */
    double getSample()
    {
        rp            = constrain(wp, buffer.size());
        double sample = buffer(rp);
        rp++;
        return sample;
    }

  private:
    Eigen::ArrayXd buffer; ///< The buffer storage for scalar samples.
    size_t         rp;     ///< The read pointer.
    size_t         wp;     ///< The write pointer.
};
