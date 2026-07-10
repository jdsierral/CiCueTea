//
//  VectorOps.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//

#pragma once

#include <Eigen/Core>

namespace jsa {

template <typename T>
inline T nextPow2(T n) { return exp2(ceil(log2(n))); }

inline double rms(const Eigen::ArrayXd& x)
{
    return std::sqrt(x.square().mean());
}

inline Eigen::ArrayXd logChirp(const Eigen::ArrayXd& t, double f0, double f1)
{
    double         t1 = t(t.size() - 1);
    Eigen::ArrayXd r =
        t.unaryExpr([&](double val) { return pow(f1 / f0, val / t1); });
    Eigen::ArrayXd phase = (t1 / std::log(f1 / f0) * f0) * (r - 1.0);
    return (phase * 2 * M_PI).cos(); // Chirp signal
}

inline std::vector<std::vector<double>>
toStdVector(const Eigen::ArrayXXd& arr)
{
    std::vector<std::vector<double>> vec(arr.rows(),
                                         std::vector<double>(arr.cols()));

    for (Eigen::Index i = 0; i < arr.rows(); ++i) {
        for (Eigen::Index j = 0; j < arr.cols(); ++j) {
            vec[i][j] = arr(i, j); // Row-major filling
        }
    }
    return vec;
}

template <typename Derived>
inline Derived cumsum(const Eigen::DenseBase<Derived>& input)
{
    Derived result(input.size());
    if (input.size() == 0)
        return result;

    result(0) = input(0);
    for (int i = 1; i < input.size(); ++i) {
        result(i) = result(i - 1) + input(i);
    }
    return result;
}

} // namespace jsa
