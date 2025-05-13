//
//  RingBuffer.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/23/25.
//

#pragma once

#include <Eigen/Core>

/**
 * @namespace jsa
 * @brief Contains utility functions and classes for buffer management and mathematical operations.
 */

/**
 * @brief Computes the next power of 2 greater than or equal to the given number.
 * @param x The input number.
 * @return The next power of 2 greater than or equal to x.
 */
inline constexpr uint32_t nextPow2(uint32_t x);

/**
 * @brief Constrains an index to wrap around within the bounds of a given size.
 * @param idx The index to constrain.
 * @param size The size of the range.
 * @return The constrained index.
 */
inline size_t constrain(size_t idx, size_t size);

/**
 * @class DoubleBuffer
 * @brief A double-buffer implementation for storing and switching between two values.
 * @tparam T The type of the values stored in the buffer.
 */
template <typename T>
class DoubleBuffer {
public:
    /**
     * @brief Fills both buffers with the given value.
     * @param value The value to fill the buffers with.
     */
    void fill(T& value);

    /**
     * @brief Pushes a value into the buffer and advances the state.
     * @param value The value to push into the buffer.
     */
    void push(T& value);

    /**
     * @brief Advances the buffer state to the next buffer.
     */
    void advance();

    /**
     * @brief Advances the buffer state and retrieves the next buffer value.
     * @return A reference to the next buffer value.
     */
    T& next();

    /**
     * @brief Retrieves the current buffer value.
     * @return A const reference to the current buffer value.
     */
    const T& current() const;

    /**
     * @brief Retrieves the last buffer value.
     * @return A const reference to the last buffer value.
     */
    const T& last() const;

    /**
     * @brief Retrieves the last buffer value.
     * @return A reference to the last buffer value.
     */
    T& last();

    /**
     * @brief Retrieves the current buffer value.
     * @return A reference to the current buffer value.
     */
    T& current();

private:
    std::array<T, 2> buffer; ///< The double buffer storage.
    bool state = false;      ///< The current state of the buffer (true or false).
};

/**
 * @class MatBuffer
 * @brief A ring buffer for storing and retrieving matrices.
 */
class MatBuffer {
public:
    /**
     * @brief Sets the size of the buffer to the next power of 2 and initializes it with a zero matrix.
     * @param newSize The desired size of the buffer.
     * @param zero The zero matrix used for initialization.
     */
    void setSize(Eigen::Index newSize, const Eigen::ArrayXXd& zero);

    /**
     * @brief Pushes a matrix into the buffer.
     * @param mat The matrix to push into the buffer.
     */
    void pushMat(const Eigen::ArrayXd& mat);

    /**
     * @brief Retrieves a matrix from the buffer.
     * @return A mapped constant reference to the retrieved matrix.
     */
    Eigen::Map<const Eigen::ArrayXXd> getMat();

private:
    std::vector<Eigen::ArrayXXd> buffer; ///< The buffer storage for matrices.
    size_t rp; ///< The read pointer.
    size_t wp; ///< The write pointer.
};

/**
 * @class BlockBuffer
 * @brief A ring buffer for storing and retrieving blocks of data.
 */
class BlockBuffer {
public:
    /**
     * @brief Sets the size of the buffer to the next power of 2 and initializes it with a zero block.
     * @param newSize The desired size of the buffer.
     * @param zero The zero block used for initialization.
     */
    void setSize(Eigen::Index newSize, const Eigen::ArrayXd& zero);

    /**
     * @brief Pushes a block of data into the buffer.
     * @param block The block of data to push into the buffer.
     */
    void pushBlock(const Eigen::ArrayXd& block);

    /**
     * @brief Retrieves a block of data from the buffer.
     * @return A mapped constant reference to the retrieved block.
     */
    Eigen::Map<const Eigen::ArrayXd> getBlock();

private:
    Eigen::ArrayXXd buffer; ///< The buffer storage for blocks of data.
    size_t rp; ///< The read pointer.
    size_t wp; ///< The write pointer.
};

/**
 * @class RingBuffer
 * @brief A ring buffer for storing and retrieving scalar samples.
 */
class RingBuffer {
public:
    /**
     * @brief Sets the size of the buffer to the next power of 2.
     * @param newSize The desired size of the buffer.
     */
    void setSize(Eigen::Index newSize);

    /**
     * @brief Pushes a scalar sample into the buffer.
     * @param sample The scalar sample to push into the buffer.
     */
    void pushSample(double sample);

    /**
     * @brief Retrieves a scalar sample from the buffer.
     * @return The retrieved scalar sample.
     */
    double getSample();

private:
    Eigen::ArrayXd buffer; ///< The buffer storage for scalar samples.
    size_t rp; ///< The read pointer.
    size_t wp; ///< The write pointer.
};