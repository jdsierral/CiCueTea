//
//  TestSignals.h
//  CQTDSP
//
//  Created by Juan Sierra on 3/17/25.
//
//  Test-only signal generation and metrics. Lives in jsa::cicuetea::test so helpers
//  can never collide with library symbols in jsa::cicuetea (e.g. MathUtils.h).
//

#pragma once

#include <cmath>
#include <numbers>

#include <Eigen/Core>

namespace jsa::cicuetea::test {

/// Root-mean-square of a signal; the reconstruction-error metric used
/// throughout the test suite.
inline double rms(const Eigen::ArrayXd& x)
{
    return std::sqrt(x.square().mean());
}

/// Exponential (log-frequency) chirp from f0 to f1 over the time axis t.
inline Eigen::ArrayXd logChirp(const Eigen::ArrayXd& t, double f0, double f1)
{
    double         t1 = t(t.size() - 1);
    Eigen::ArrayXd r =
        t.unaryExpr([&](double val) { return pow(f1 / f0, val / t1); });
    Eigen::ArrayXd phase = (t1 / std::log(f1 / f0) * f0) * (r - 1.0);
    return (phase * 2 * std::numbers::pi).cos(); // Chirp signal
}

} // namespace jsa::cicuetea::test
