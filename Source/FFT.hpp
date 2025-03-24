//
//  FFT.hpp
//  CQTDSP
//
//  Created by Juan Sierra on 3/9/25.
//

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

namespace jsa {

class DFTImpl;

class DFT {
public:
    DFT();
    void init(size_t fftSize);
    
    void dft(const Eigen::ArrayXcd& X, Eigen::ArrayXcd& Y);
    void idft(const Eigen::ArrayXcd& X, Eigen::ArrayXcd& Y);
    void rdft(const Eigen::ArrayXd& x, Eigen::ArrayXcd& X);
    void irdft(const Eigen::ArrayXcd& X, Eigen::ArrayXd& x);
    
    void dft(const Eigen::ArrayXXcd& X, Eigen::ArrayXXcd& Y);
    void idft(const Eigen::ArrayXXcd& X, Eigen::ArrayXXcd& Y);
    void rdft(const Eigen::ArrayXXd& x, Eigen::ArrayXXcd& X);
    void irdft(const Eigen::ArrayXXcd& X, Eigen::ArrayXXd& x);
    
private:
    DFTImpl* pImpl;
};

}
