//
//  MathUtils.h
//  CQTDSP
//
//  Created by Juan Sierra on 5/1/25.
//

#pragma once

#include <Eigen/Core>

namespace jsa {

/**
 * @brief Computes the next power of 2 greater than or equal to the given
 * number.
 *
 * This function calculates the smallest power of 2 that is greater than or
 * equal to the input value `x`. If `x` is 0, the function returns 1.
 *
 * @param x The input value for which the next power of 2 is to be computed.
 *          Must be a 32-bit unsigned integer.
 * @return The next power of 2 greater than or equal to `x`.
 *
 * @note This function uses bitwise operations to efficiently compute the
 * result. It assumes that the input is within the range of a 32-bit unsigned
 * integer.
 */
inline constexpr uint32_t nextPow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

/**
 * @brief Constrains an index to fit within the bounds of a given size by
 * wrapping it around.
 *
 * This function ensures that the provided index remains within the range [0,
 * size - 1] by using the modulo operation. It is useful for cyclic or circular
 * indexing.
 *
 * @param idx The index to be constrained.
 * @param size The size of the range within which the index should be
 * constrained.
 * @return The constrained index within the range [0, size - 1].
 */
inline size_t constrain(size_t idx, size_t size) { return idx % size; }

/**
 * @brief Computes the square of a given number.
 *
 * This function takes a double-precision floating-point number as input
 * and returns its square (the number multiplied by itself).
 *
 * @param x The number to be squared.
 * @return The square of the input number.
 */
inline double square(double x) { return x * x; }

/**
 * @brief Generates a linearly spaced array of values from 0 to num-1.
 *
 * @param num The number of elements in the resulting array.
 * @return Eigen::ArrayXd A 1D array containing linearly spaced values from 0 to
 * num-1.
 */
inline Eigen::ArrayXd regspace(Eigen::Index num) {
    return Eigen::ArrayXd::LinSpaced(num, 0, num - 1);
}

/**
 * @brief Generates a linearly spaced array of values between a specified range.
 *
 * This function creates an Eigen::ArrayXd containing evenly spaced values
 * starting from `low` and ending at `high`, inclusive. The number of elements
 * in the array is determined by the difference between `high` and `low` plus
 * one.
 *
 * @param low The starting value of the range (inclusive).
 * @param high The ending value of the range (inclusive).
 * @return Eigen::ArrayXd A 1D array of linearly spaced values from `low` to
 * `high`.
 */
inline Eigen::ArrayXd regspace(Eigen::Index low, Eigen::Index high) {
    return Eigen::ArrayXd::LinSpaced(high - low + 1, low, high);
}

/**
 * @brief Generates a logarithmically spaced array.
 *
 * This function creates an array of `num` elements, logarithmically spaced
 * between `start` and `end`. The values are computed using the natural
 * logarithm and then exponentiated to return the final array.
 *
 * @param start The starting value of the sequence (must be positive).
 * @param end The ending value of the sequence (must be positive).
 * @param num The number of elements in the sequence.
 * @return Eigen::ArrayXd A logarithmically spaced array of size `num`.
 *
 * @note The `start` and `end` parameters must be greater than zero, as the
 *       logarithm of non-positive numbers is undefined.
 */
inline Eigen::ArrayXd logspace(double start, double end, Eigen::Index num) {
    return Eigen::ArrayXd::LinSpaced(num, log(start), log(end)).exp();
}

}
