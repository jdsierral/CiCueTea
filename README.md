# 🍵 CiCueTea

**CiCueTea** *(pronounced "C-Q-T")* is a real-time, invertible **Constant-Q Transform (CQT)** engine based on **nonstationary Gabor frames**, built for low-latency spectral signal processing. It powers the core of the [CiCueProc](https://JuanSaudio.com/audio-plugins) plugin suite.

> 🎧 “Brew your spectrum.”

---

## Features

- **Real-time**: designed for the audio callback — suitable for plugins, interactive DSP, and other low-latency environments.
- **Invertible**: forward/inverse reconstruction to near machine precision (~10⁻¹⁶, verified by the unit tests).
- **Constant-Q resolution**: high frequency resolution at low frequencies, high time resolution at high frequencies.
- **Pitch-symmetric**: Gaussian windows designed in log-frequency give passbands that are symmetric in pitch, not just in Hz.
- **Two variants**: a *dense* version with the same sample rate in every band, and a *sparse* version with a decimated per-band sample rate.
- **Multiple FFT backends**: vDSP (default on macOS), MKL, FFTW, and PFFFT.
- **Reference implementations** in MATLAB and Python are included.

> **Note:** FFTW is GPL-licensed — building CiCueTea against the FFTW backend subjects the resulting binary to the GPL. The other backends carry no such restriction.

---

## Requirements

- A **C++20** compiler
- **CMake ≥ 3.22** (Ubuntu 22.04 LTS stock)
- **Eigen ≥ 3.4** (Eigen 5.x supported)
- An FFT backend, selected via `FFTSelection.cmake` with per-platform defaults: **vDSP** (macOS, system-provided), **MKL** (Windows), **FFTW** (Linux — `apt install libfftw3-dev`; note FFTW is GPL), or **PFFFT** anywhere with `-DFFT_PFFFT=ON`
- **Boost ≥ 1.70**, headers only (unit tests only — the library itself has no Boost dependency)

---

## Installation

### CMake (recommended)

```cmake
# Top-level CMakeLists.txt
add_subdirectory(Libs/CiCueTea)

target_link_libraries(MyPlugin PRIVATE CiCueTea)
target_include_directories(MyPlugin PRIVATE Libs/CiCueTea/include)
```

Or link it as a Git submodule:

```bash
git submodule add https://github.com/jdsierral/CiCueTea Libs/CiCueTea
git submodule update --init --recursive
```

---

## Example Usage

### Whole-buffer transform

The transform objects themselves work on a full buffer — useful for analysis,
offline processing, and understanding the parameters:

```cpp
#include <Eigen/Core>
#include <CQT.hpp>

long   nSamps = 1<<16;
double fs     = 48000;
double frac   = 1.0/48.0;  // 48 bands per octave
double fMin   = 100;
double fMax   = 10000;
double fRef   = 440;

jsa::cicuetea::NsgfCqtDense cqt(fs, nSamps, frac, fMin, fMax, fRef);

Eigen::ArrayXd   x(cqt.getNumSamples());
Eigen::ArrayXd   y(cqt.getNumSamples());
Eigen::ArrayXXcd Xcq(cqt.getNumSamples(), cqt.getNumBands());

cqt.forward(x, Xcq);   // Forward transform
cqt.inverse(Xcq, y);   // Inverse transform
```

The sparse variant differs only in construction and coefficient storage:

```cpp
jsa::cicuetea::NsgfCqtSparse cqt(fs, nSamps, frac, fMin, fMax, fRef);

Eigen::ArrayXd x(cqt.getNumSamples());
Eigen::ArrayXd y(cqt.getNumSamples());
auto Xcq = cqt.getCoefs();

cqt.forward(x, Xcq);
cqt.inverse(Xcq, y);
```

### Real-time streaming

The whole point of CiCueTea is that the above also runs **inside an audio
callback**. The processor classes in `CQTProcessor.hpp` wrap a transform in a
slice → forward → *your processing* → inverse → overlap-add chain: derive from
one, put your spectral manipulation in `processBlock()` (it receives the CQT
coefficients of one block, to be modified in place), and drive it one sample at
a time:

```cpp
#include <CQTProcessor.hpp>

using namespace jsa::cicuetea;

class LowBandGain : public SlidingCqtSparseProcessor
{
  public:
    using SlidingCqtSparseProcessor::SlidingCqtSparseProcessor;

    void processBlock(NsgfCqtSparse::Coefs& Xcq) override
    {
        // Xcq[k] is band k's complex coefficient series (decimated per band).
        for (size_t k = 0; k < Xcq.size() / 2; k++)
            Xcq[k] *= 0.5;  // -6 dB below the spectral midpoint
    }
};

long   nSamps = 1<<14;     // blockSize
double fs     = 48000;
double frac   = 1.0/12.0;  // 12 bands per octave
double fMin   = 100;
double fMax   = 10000;
double fRef   = 440;

// Setup (e.g. prepareToPlay). nSamps is the internal blockSize (must be long
// enough to resolve the lowest band), and it sets the latency.
LowBandGain proc(fs, nSamps, frac, fMin, fMax, fRef);
if (!proc.isValid())
    return;  // configuration rejected — the processor is inert (see below)

// Audio callback — allocation-free by construction:
for (int n = 0; n < numSamples; n++)
    out[n] = proc.processSample(in[n]);
```

The output is the processed input delayed by exactly `getLatency()` samples —
report that to the host for plugin-delay compensation. An empty
`processBlock()` is the identity: the input comes back out at reconstruction
precision (this is how the unit tests verify the chain).

There are two processor families, each in a dense and a sparse variant:

- **`Cqt{Dense,Sparse}Processor`** — consecutive blocks, each transformed and
  inverted independently (latency = block size). Machine-exact reconstruction,
  but a coefficient manipulation that *changes over time* can jump at block
  boundaries.
- **`SlidingCqt{Dense,Sparse}Processor`** — the sliCQ-style path: overlapped,
  windowed slices whose modifications cross-fade smoothly across block
  boundaries (latency = block size + hop). The overlap windowing trades a
  small reconstruction error (~-60 dB at the example's block size, shrinking
  as the block grows) for that smoothness. This is the one for time-varying
  processing — and the piece most other libraries are missing.

`isValid()` reports whether the configuration passed the frame-health check
(e.g. a block too short to resolve `minFrequency` is rejected); an invalid
processor is inert and outputs silence rather than misbehaving.

---

## Parameters & Design Notes

| Parameter      | Description                                                                |
| -------------- | -------------------------------------------------------------------------- |
| `fs`           | Sample rate — in a CQT the design is tied to it                            |
| `nSamples`     | Number of samples to transform (for the streaming processors: the internal analysis block size, which sets the latency) |
| `frac`         | Reciprocal of bands per octave; fractional values allowed                  |
| `minFrequency` | Start of frequency range (going too low increases latency)                 |
| `maxFrequency` | Upper limit of the transform (bounds the range with the Constant-Q property) |

> CiCueTea uses **Gaussian windows designed in log-frequency** to obtain perfect pitch symmetry.

---

## How It Compares

Existing CQT implementations each offer some, but not all, of: formal invertibility, real-time-safe streaming, and log-symmetric passbands.
Measured round-trip errors and timings backing this section live in [Benchmarks/](Benchmarks/), together with a reproducible comparison script.

- [librosa](https://librosa.org) (Python) — several CQT algorithms, none NSGF-based, so none is formally invertible. An inverse (`icqt`) exists via least-squares reconstruction, but with default parameters the high frequencies are undersampled and reconstruction is approximate at best. Offline analysis only.
- [nnAudio / nnAudio2](https://github.com/AMAAI-Lab/nnAudio2) (Python/PyTorch) — GPU-accelerated, differentiable CQT layers for machine-learning pipelines; kernel-based rather than NSGF-based. nnAudio2's inverse (`iCQT`) is an iterative Landweber reconstruction — useful for training loops, but its hop-decimated frames sit below critical sampling, so broadband reconstruction is structurally out of reach (measured in [Benchmarks/](Benchmarks/)).
- TensorFlow — no native CQT; community implementations are *pseudo*-CQTs built by pooling STFT bins, which inherit the STFT’s linear-frequency resolution and offer no inverse at all.
- [cqt-pytorch](https://github.com/archinetai/cqt-pytorch) (Python/PyTorch) — NSGF-based in principle, but the code appears unmaintained and out of date; and as Python it is not suited for real-time use.
- [NSGT](https://github.com/grrrr/nsgt) (Python) — the reference NSGF implementation; formally invertible, but offline — no streaming support.
- [LTFAT](https://ltfat.org) (MATLAB/Octave) — invertible NSGF transforms including a block-streaming variant, but MATLAB is not a language for real-time deployment.
- [rt-cqt](https://github.com/jmerkt/rt-cqt) (C++) — header-only and aimed at real-time use, but kernel-based rather than NSGF-based, so not formally invertible.
- [The Gaborator](https://www.gaborator.com) (C++) — invertible and streaming-capable, but it allocates inside the processing pipeline, which rules out real-time audio callbacks. AGPL-licensed, whereas CiCueTea is MIT.
- [Essentia](https://essentia.upf.edu) (C++) — NSGF-based and invertible (`NSGConstantQ`/`NSGIConstantQ`); the closest relative. However, its “streaming” mode exists only as a Python-side wrapper over independent blocks — the C++ core has no streaming path — and it provides no bridge from the long-form representation (CQ-NSGT) to a sliced, overlapped one (sliCQ-style), so seamless block-wise reconstruction is not possible. Windows are Hann-family in linear frequency; aimed at offline feature extraction rather than real-time processing.

CiCueTea targets the full intersection: allocation-free real-time forward/inverse streaming, reconstruction to numerical precision, and — unique among these — **Gaussian windows designed in log-frequency**, giving passbands that are exactly symmetric in pitch.

---

## Building & Running the Tests

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build -j4 --output-on-failure
```

The test suite covers the DFT backends, dense/sparse forward-inverse round trips (reconstruction error ≈ 3×10⁻¹⁶), the sliding CQT, and the slicing/splicing machinery. Benchmark-style tests are labeled `bench`; skip them for a quick correctness run with `ctest -LE bench`.

---

## Used in

[`CiCueEq`](https://JuanSaudio.com/audio-plugins/CiCueEq) ·
[`CiCueDenoise`](https://JuanSaudio.com/audio-plugins/CiCueDenoise) ·
[`CiCueDecorr`](https://JuanSaudio.com/audio-plugins/CiCueDecorr) ·
[`PitchDelay`](https://JuanSaudio.com/audio-plugins/PitchDelay) ·
[`PitchScrambler`](https://JuanSaudio.com/audio-plugins/PitchScrambler) ·
[`PitchFDN`](https://JuanSaudio.com/audio-plugins/PitchFDN)

---

## Name?

**Ci·Cue·Tea** → say it letter by letter: **“C-Q-T”**.

A spectral engine so smooth, you’ll want a second cup.

---

## Citation

If you use CiCueTea in academic work, please cite it — see [`CITATION.cff`](CITATION.cff) or use GitHub’s *“Cite this repository”* button. Papers describing the underlying frame design and the library are in preparation.

---

## References

The theory behind CiCueTea — the pitch-symmetric log-Gaussian frame design and the real-time implementation strategy — is developed in:

- J. Sierra, *[Constant-Q Spectral Signal Processing](https://www.proquest.com/openview/ffa6564d1fea773073832212b08bf0e1/1?pq-origsite=gscholar&cbl=18750&diss=y)*, Ph.D. dissertation, New York University, 2025.

Foundational literature on nonstationary Gabor frames and the invertible CQT:

- P. Balazs, M. Dörfler, F. Jaillet, N. Holighaus, and G. A. Velasco, “Theory, implementation and applications of nonstationary Gabor frames,” *J. Comput. Appl. Math.*, 236(6), 2011.
- G. A. Velasco, N. Holighaus, M. Dörfler, and T. Grill, “Constructing an invertible constant-Q transform with nonstationary Gabor frames,” *Proc. DAFx-11*, 2011.
- N. Holighaus, M. Dörfler, G. A. Velasco, and T. Grill, “A framework for invertible, real-time constant-Q transforms,” *IEEE Trans. Audio, Speech, Lang. Process.*, 21(4), 2013.

---

## License

[MIT License](LICENSE) — use it freely, sip responsibly.

---

## Author

Developed by [Juan Sierra](https://github.com/jdsierral) as part of research at NYU and NYU Abu Dhabi.
Website: [JuanSaudio](https://JuanSaudio.com)
