# 🍵 CiCueTea

**CiCueTea** is a high-performance, real-time Constant-Q Transform engine based on **nonstationary Gabor frames**. Built for spectral signal processing. It is designed to be invertibile and low-latency operation, it powers the core of the [CiCueProc](www.JuanSaudio.com/audio-plugins) plugin suite.

> 🎧 “Brew your spectrum.”™



---

## ✨ Features

- ⚡ **Real-time performance**: Suitable for plugin use, interactive DSP, and low-latency environments.
- ♻ **Invertibile** (within numerical tolerance): Enables seamless reconstruction after transformation.
- 🔍 **High frequency resolution** at low frequencies, **high time resolution** at high frequencies.
- 🧠 **Based on Nonstationary Gabor Frames (NSGF)**: Sample-exact theoretical foundation.
- 🛠️ Modular design: Drop into any C++ project

---

## 🔧 Installation

### CMake (recommended)

```cmake
# Top-level CMakeLists.txt
add_subdirectory(Libs/CiCueTea)

target_link_libraries(MyPlugin PRIVATE CiCueTea)
target_include_directories(MyPlugin PRIVATE Libs/CiCueTea/include)
```

Or link it as a Git submodule:

```bash
git submodule add https://github.com/your-org/CiCueTea Libs/CiCueTea
git submodule update --init --recursive
```

---

## 🧪 Example Usage

```cpp
#include <Eigen/Core>
#include <CQT.hpp>

double fs = 48000;
long N = 1<<16;
double fMin = 100;
double fMax = 10000;
double fRef = 440;
jsa::NsgfCqtDense cqt(fs, nSamps, frac, fMin, fMax, fRef);

Eigen::ArrayXd x(cqt.getNumSamples());
Eigen::ArrayXd y(cqt.getNumSamples());
Eigen::ArrayXXcd Xcq(cqt.getNumSamples(), cqt.getNumBands());

// Forward transform
cqt.forward(x, Xcq);

// Inverse transform
cqt.inverse(Xcq, y);
```

---

## 📀 Parameters & Design Notes

| Parameter       | Description                                                 |
| --------------- | ----------------------------------------------------------- |
| `fs`            | Sample Rate since in the CQT it is highly connected         |
| `nSamples`      | Number of Samples to transform                              |
| `frac`          | This is the reciprocal of points per octave allowing        | 
|                 | fractional values                                           |
| `minFrequency`  | Start of frequency range (e.g., 100 Hz as going to low      |
|                 | increases latency)                                          |
| `maxFrequency`  | Upper limit of transform (Limits the range with             |
|                 |Constant-Q property)                                         |
| --------------- | ----------------------------------------------------------- |

> CiCueTea uses **Gaussian windows designed in log-frequency** to obtain perfect
> pitch symmetry.

---

## 🧠 What Makes It Special?

Most CQT implementations either:

- Are not invertible,
- Are not usable in real time,
- Are not designed for true symmetric log-frequency shaped pass-bands

**CiCueTea** is designed to achieve **all three**:

- Real-time forward/inverse streaming
- Numerically accurate reconstruction
- High resolution in perceptually-relevant bands

---

## 📦 Used in

- 🎛️ [`CiCueEq`](www.JuanSaudio.com/audio-plugins/CiCueEq)
- 🎚️ [`CiCueDenoise`](www.JuanSaudio.com/audio-plugins/CiCueDenoise)
- 🔊 [`CiCueDecorr`](www.JuanSaudio.com/audio-plugins/CiCueDecorr)
- 🎛️ [`PitchDelay`](www.JuanSaudio.com/audio-plugins/PitchDelay)
- 🎚️ [`PitchScrambler`](www.JuanSaudio.com/audio-plugins/PitchScrambler)
- 🔊 [`PitchFDN`](www.JuanSaudio.com/audio-plugins/PitchFDN)


---

## 🧳️ Name?

**CiCueTea** → "CQT"

A spectral engine so smooth, you’ll want a second cup.

---

## 🪖 License

MIT License — use it freely, sip responsibly.

---

## 👤 Author

Developed by [Juan Sierra](https://github.com/jdsierral) as part of research at NYU Abu Dhabi.

---

## 🧪 Advanced Options

Want to explore field separation, harmonic freezing, or log-frequency spectral effects?

CiCueTea was designed for research-driven, next-generation DSP workflows. Reach out or contribute if you'd like to help expand it.

