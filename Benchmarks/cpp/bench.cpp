//
//  bench.cpp
//  CiCueTea Benchmarks
//
//  C++ counterpart of ../compare.py: round-trip reconstruction error and
//  forward/inverse wall time on the same task (white noise, 2^20 samples at
//  48 kHz, 12 bands/octave), best of 3 runs, printed as a markdown table with
//  a self-documenting environment report.
//
//  The Gaborator comparison is compiled in only when the build is pointed at
//  a locally downloaded copy (-DGABORATOR_DIR=...); it is AGPLv3/commercial,
//  so this repository neither vendors nor depends on it. See ../README.md.
//

#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <CQT.hpp>

#ifdef HAVE_GABORATOR
#include <gaborator/gaborator.h>
#include <gaborator/version.h>
#endif

#ifdef HAVE_RTCQT
#include <ConstantQTransform.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

using namespace Eigen;
using namespace jsa::cicuetea;

namespace {

constexpr double fs      = 48000.0;
constexpr double frac    = 1.0 / 12;
constexpr double fMin    = 100.0;
constexpr double fMax    = 10000.0;
constexpr double fRef    = 1000.0;
constexpr int    repeats = 3;

double rms(const ArrayXd& a, const ArrayXd& b) { return std::sqrt((a - b).square().mean()); }

struct Row {
    std::string name, config;
    double      err, fwdMs, invMs; // invMs < 0: interleaved API, total under fwdMs
    long long   nCoefs = -1;       // stored complex values; -1: not measured
};

/// Best-of-N timing of a forward/inverse pair; error from the last run.
template <typename Fwd, typename Inv>
Row timeRoundTrip(std::string name, std::string config, const ArrayXd& x,
                  ArrayXd& y, Fwd&& fwd, Inv&& inv, long long nCoefs)
{
    using clk    = std::chrono::steady_clock;
    double bestF = 0, bestI = 0;
    for (int r = 0; r < repeats; r++) {
        auto t0 = clk::now();
        fwd();
        auto t1 = clk::now();
        inv();
        auto   t2 = clk::now();
        double f  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double i  = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (r == 0 || f + i < bestF + bestI) { bestF = f; bestI = i; }
    }
    return {std::move(name), std::move(config), rms(x, y), bestF, bestI, nCoefs};
}

std::string cpuName()
{
#ifdef __APPLE__
    char   buf[256];
    size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
        return buf;
#endif
    return "unknown";
}

std::string compilerName()
{
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

} // namespace

int main()
{
    const Index n = Index(1) << 20;

    std::cout << "## CQT C++ benchmark\n\n```\n";
    std::time_t now = std::time(nullptr);
    char        date[16];
    std::strftime(date, sizeof(date), "%Y-%m-%d", std::localtime(&now));
    std::cout << "date:     " << date << "\n"
              << "machine:  " << cpuName() << "\n"
              << "compiler: " << compilerName() << "\n"
              << "eigen:    " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION
              << "." << EIGEN_MINOR_VERSION << "\n"
#ifdef HAVE_GABORATOR
              << "gaborator:" << " " << GABORATOR_VERSION_MAJOR << "."
              << GABORATOR_VERSION_MINOR << "\n"
#else
              << "gaborator: not built (set -DGABORATOR_DIR to include it)\n"
#endif
              << "signal:   white noise, N = 2^20 = " << n << " samples @ 48000 Hz ("
              << double(n) / fs << " s), best of " << repeats << " runs\n"
              << "```\n\n";

    ArrayXd x(n), y(n);
    {
        std::mt19937_64                  gen(42);
        std::normal_distribution<double> dist;
        for (Index i = 0; i < n; i++) x(i) = dist(gen);
    }

    std::vector<Row> rows;

    {
        NsgfCqtDense cqt(fs, n, frac, fMin, fMax, fRef);
        ArrayXXcd    Xcq(cqt.getNumSamps(), cqt.getNumBands());
        rows.push_back(timeRoundTrip(
            "CiCueTea (dense)",
            "dense, " + std::to_string(cqt.getNumBands()) + " bands, 100-10000 Hz",
            x, y, [&] { cqt.forward(x, Xcq); }, [&] { cqt.inverse(Xcq, y); },
            (long long)Xcq.size()));
    }
    {
        NsgfCqtSparse cqt(fs, n, frac, fMin, fMax, fRef);
        auto          Xcq = cqt.getCoefs();
        long long     nCoefs = 0;
        for (const auto& c : Xcq) nCoefs += (long long)c.size();
        rows.push_back(timeRoundTrip(
            "CiCueTea (sparse)",
            "sparse, " + std::to_string(cqt.getNumBands()) + " bands, 100-10000 Hz",
            x, y, [&] { cqt.forward(x, Xcq); }, [&] { cqt.inverse(Xcq, y); },
            nCoefs));
    }

#ifdef HAVE_GABORATOR
    {
        gaborator::log_fq_scale     scale(1.0 / frac, fMin / fs);
        gaborator::parameters       params(scale);
        gaborator::analyzer<double> analyzer(params);
        // analyze() accumulates into coefs, so use a fresh container per run
        // (allocation happens lazily inside analyze; both libraries pay their
        // own coefficient-storage cost inside the timed region or before it —
        // noted in ../README.md).
        std::unique_ptr<gaborator::coefs<double>> coefs;
        auto row = timeRoundTrip(
            "Gaborator",
            std::to_string(int(1.0 / frac)) + " bpo, " +
                std::to_string(analyzer.n_bands_total) +
                " bands, 100 Hz-Nyquist (by design), double",
            x, y,
            [&] {
                coefs = std::make_unique<gaborator::coefs<double>>(analyzer);
                analyzer.analyze(x.data(), 0, n, *coefs);
            },
            [&] { analyzer.synthesize(*coefs, 0, n, y.data()); }, -1);
        // Count every stored coefficient (they extend past [0, n) by the
        // atoms' time support — that is its true storage footprint).
        long long cnt = 0;
        gaborator::process(
            [&](int, int64_t, std::complex<double>&) { cnt++; },
            INT_MIN, INT_MAX, INT64_MIN, INT64_MAX, *coefs);
        row.nCoefs = cnt;
        rows.push_back(std::move(row));
    }
#endif

#ifdef HAVE_RTCQT
    {
        // Based on Juan's adapted rt-cqt example (BenchTests/Cpp/rt-cqt,
        // examples/cqt.cpp): the whole signal is pushed as one block and the
        // schedule interleaves forward and inverse per band/hop, so only the
        // combined time is measurable.
        constexpr int bpo = 12, nOct = 8; // compile-time in rt-cqt's API
        constexpr int blockSize = 1024;   // its designed streaming granularity

        using clk   = std::chrono::steady_clock;
        double best = 0;
        for (int r = 0; r < repeats; r++) {
            // Fresh instance per run (untimed): rt-cqt's internal circular
            // buffers accumulate across calls, so it cannot be re-run on the
            // same instance without reinitialization.
            auto cqt = std::make_unique<Cqt::ConstantQTransform<bpo, nOct>>();
            cqt->init(256); // hop size
            cqt->initFs(fs, blockSize);

            auto t0 = clk::now();
            for (Index i0 = 0; i0 + blockSize <= n; i0 += blockSize) {
                cqt->inputBlock(x.data() + i0, blockSize);
                for (const auto& s : cqt->getCqtSchedule()) {
                    cqt->cqt(s);
                    cqt->icqt(s);
                }
                auto* out = cqt->outputBlock(blockSize);
                for (Index i = 0; i < blockSize; i++) y(i0 + i) = out[i];
            }
            auto   t1 = clk::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (r == 0 || ms < best) best = ms;
        }

        // Score at the best *global* alignment (circular cross-correlation).
        // No such alignment recovers the waveform: rt-cqt's multirate pipeline
        // gives each octave a different latency, so the reconstruction is
        // per-band dispersed by design — the measured error stays at signal
        // level however the output is shifted. See ../README.md.
        DFT      dftN{size_t(n)};
        ArrayXcd X = ArrayXcd::Zero(n), Y = ArrayXcd::Zero(n);
        ArrayXd  r(n);
        dftN.rdft(x, X);
        dftN.rdft(y, Y);
        X = X.conjugate() * Y;
        dftN.irdft(X, r);
        Index lag = 0;
        r.abs().maxCoeff(&lag);
        ArrayXd yAligned(n);
        for (Index i = 0; i < n; i++) yAligned(i) = y((i + lag) % n);

        rows.push_back({"rt-cqt",
                        std::to_string(bpo * nOct) +
                            " bins, 8 octaves below Nyquist, hop 256, streamed "
                            "1024-sample blocks, best global alignment",
                        rms(x, yAligned), best, -1.0, -1});
    }
#endif

    auto fmtErr = [](double e) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.2e", e);
        return std::string(b);
    };
    auto fmtMs = [](double m) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.1f", m);
        return std::string(b);
    };
    auto fmtXrt = [n](double f, double i) {
        double x = (double(n) / fs) * 1e3 / (f + std::max(i, 0.0));
        char   b[32];
        std::snprintf(b, sizeof(b), x < 10 ? "%.1fx" : "%.0fx", x);
        return std::string(b);
    };

    auto fmtCoefs = [n](long long c) {
        if (c < 0) return std::string("n/a");
        char b[48];
        std::snprintf(b, sizeof(b), "%.2e (%.2fx)", double(c), double(c) / double(n));
        return std::string(b);
    };

    std::vector<std::vector<std::string>> cells = {
        {"Implementation", "Configuration", "Round-trip RMS error",
         "Coefficients (xN)", "Forward (ms)", "Inverse (ms)", "x realtime"}};
    for (auto& r : rows)
        cells.push_back({r.name, r.config, fmtErr(r.err), fmtCoefs(r.nCoefs),
                         r.invMs < 0 ? fmtMs(r.fwdMs) + " (fwd+inv)" : fmtMs(r.fwdMs),
                         r.invMs < 0 ? "n/a" : fmtMs(r.invMs),
                         fmtXrt(r.fwdMs, r.invMs)});

    constexpr size_t    nCols = 7;
    std::vector<size_t> w(nCols, 0);
    for (auto& row : cells)
        for (size_t c = 0; c < nCols; c++) w[c] = std::max(w[c], row[c].size());
    for (size_t i = 0; i < cells.size(); i++) {
        std::cout << "|";
        for (size_t c = 0; c < nCols; c++)
            std::cout << " " << cells[i][c]
                      << std::string(w[c] - cells[i][c].size(), ' ') << " |";
        std::cout << "\n";
        if (i == 0) {
            std::cout << "|";
            for (size_t c = 0; c < nCols; c++)
                std::cout << std::string(w[c] + 2, '-') << "|";
            std::cout << "\n";
        }
    }
    return 0;
}
