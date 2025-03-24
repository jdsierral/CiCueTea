//
//  VectorOps.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <Eigen/Core>
#include <armadillo>

namespace jsa {

inline double square(double x) { return x * x; }

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

inline size_t constrain(size_t idx, size_t size) {
    return idx % size;
}

template <typename T> inline T nextPow2(T n) { return exp2(ceil(log2(n))); }

inline arma::vec eig2armaVec(Eigen::ArrayXd x) {
    return arma::vec(x.data(), x.size());
}

inline arma::cx_vec eig2armaVec(Eigen::ArrayXcd x) {
    return arma::cx_vec(x.data(), x.size());
}

inline arma::mat eig2armaMat(Eigen::ArrayXXd x) {
    return arma::mat(x.data(), x.rows(), x.cols());
}

inline arma::cx_mat eig2armaMat(Eigen::ArrayXXcd x) {
    return arma::cx_mat(x.data(), x.rows(), x.cols());
}

inline double rms(const Eigen::ArrayXd& x) {
    return std::sqrt(x.square().mean());
}

inline Eigen::ArrayXd regspace(Eigen::Index num) {
    return Eigen::ArrayXd::LinSpaced(num, 0, num-1);
}

inline Eigen::ArrayXd regspace(Eigen::Index low, Eigen::Index high) {
    return Eigen::ArrayXd::LinSpaced(high-low+1, low, high);
}

static Eigen::ArrayXd hann(Eigen::Index N) {
    Eigen::ArrayXd n = regspace(N);
    Eigen::ArrayXd win = (M_PI * n / N).sin().square();
    return win;
}

static Eigen::ArrayXd logChirp(const Eigen::ArrayXd& t, double f0, double f1) {
    double t1 = t(t.size() - 1);
    Eigen::ArrayXd r = t.unaryExpr([&](double val){ return pow(f1/f0, val/t1); });
    Eigen::ArrayXd phase = (t1/std::log(f1/f0)*f0)*(r - 1.0);
    return (phase * 2 * M_PI).cos();  // Chirp signal
}

inline std::vector<std::vector<double>> toStdVector(const Eigen::ArrayXXd& arr) {
    std::vector<std::vector<double>> vec(arr.rows(), std::vector<double>(arr.cols()));

    for (Eigen::Index i = 0; i < arr.rows(); ++i) {
        for (Eigen::Index j = 0; j < arr.cols(); ++j) {
            vec[i][j] = arr(i, j);  // Row-major filling
        }
    }
    return vec;
}

template <typename Derived>
inline Derived cumsum(const Eigen::DenseBase<Derived>& input) {
    Derived result(input.size());
    if (input.size() == 0) return result;

    result(0) = input(0);
    for (int i = 1; i < input.size(); ++i) {
        result(i) = result(i - 1) + input(i);
    }
    return result;
}

}
