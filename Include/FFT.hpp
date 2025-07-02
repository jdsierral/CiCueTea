//
//  FFT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

/**
 * @file FFT.hpp
 * @brief Provides an implementation of Fast Fourier Transform (DFT) and its inverses.
 * @author Juan Sierra
 * @date 3/9/25
 * @copyright MIT License
 */

#pragma once

#include <Eigen/Core>

namespace jsa {

class DFTImpl;

/**
 * @class DFT
 * @ingroup SignalProcessing
 * 
 * This is a pImpl based wrapper to other commonly known FFTs like 
 * - FFTW
 * - Accelerate
 * - MKL
 * - PFFFT
 * 
 * It also provides interfaces for different "versions" of Fourier Transforms
 * like dft and idft (a complex to complex transform), and rdft and irdft (a 
 * real to complex transform). Moreover, it also provides interfaces to process
 * many DFTs when the data is based on a matrix; however it is simply calling
 * the single DFTs repeatedly
 * 
 * @brief The Wrapper class for other FFTs provided by different libraries
 */
class DFT {
public:
    
    /**
     * @brief Constructor.
     * @param fftSize FFT size (power of two).
     */
    DFT(size_t fftSize);
    
    /**
     * @brief Computes the Discrete Fourier Transform (DFT) on 1D data.
     * @param X Input array of complex values.
     * @param Y Output array of transformed complex values.
     */
    void dft(const Eigen::ArrayXcd& X, Eigen::ArrayXcd& Y);

    /**
     * @brief Computes the inverse Discrete Fourier Transform (IDFT) on 1D data.
     * @param X Input array of complex values.
     * @param Y Output array of transformed complex values.
     */
    void idft(const Eigen::ArrayXcd& X, Eigen::ArrayXcd& Y);

    /**
     * @brief Computes the forward Real Discrete Fourier Transform (RDFT) on 1D data.
     * @param x Input array of real values.
     * @param X Output array of transformed complex values.
     */
    void rdft(const Eigen::ArrayXd& x, Eigen::ArrayXcd& X);

    /**
     * @brief Computes the inverse Real Discrete Fourier Transform (IRDFT) on 1D data.
     * @param X Input array of complex values.
     * @param x Output array of transformed real values.
     */
    void irdft(const Eigen::ArrayXcd& X, Eigen::ArrayXd& x);

    /**
     * @brief Computes the forward Discrete Fourier Transform (DFT) on 2D data.
     * @param X Input matrix of complex values.
     * @param Y Output matrix of transformed complex values.
     */
    void dft(const Eigen::ArrayXXcd& X, Eigen::ArrayXXcd& Y);

    /**
     * @brief Computes the inverse Discrete Fourier Transform (IDFT) on 2D data.
     * @param X Input matrix of complex values.
     * @param Y Output matrix of transformed complex values.
     */
    void idft(const Eigen::ArrayXXcd& X, Eigen::ArrayXXcd& Y);

    /**
     * @brief Computes the forward Real Discrete Fourier Transform (RDFT) on 2D data.
     * @param x Input matrix of real values.
     * @param X Output matrix of transformed complex values.
     */
    void rdft(const Eigen::ArrayXXd& x, Eigen::ArrayXXcd& X);

    /**
     * @brief Computes the inverse Real Discrete Fourier Transform (IRDFT) on 2D data.
     * @param X Input matrix of complex values.
     * @param x Output matrix of transformed real values.
     */
    void irdft(const Eigen::ArrayXXcd& X, Eigen::ArrayXXd& x);

private:
    /**
     * @brief Pointer to the implementation of the DFT operations.
     */
    DFTImpl* pImpl;
};

}
