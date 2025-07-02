# 🍵 CiCueTea

**CiCueTea** is a high-performance, real-time Constant-Q Transform engine based on **nonstationary Gabor frames**. Built for spectral signal processing with near-perfect invertibility and low-latency operation, it powers the core of the [CiCue](https://github.com/your-org) plugin suite.

> 🎧 “Brew your spectrum.”™



---

## ✨ Features

- ⚡ **Real-time performance**: Suitable for plugin use, interactive DSP, and low-latency environments.
- ♻ **Perfect invertibility** (within numerical tolerance): Enables seamless reconstruction after transformation.
- 🔍 **High frequency resolution** at low frequencies, **high time resolution** at high frequencies.
- 🧠 **Based on Nonstationary Gabor Frames (NSGF)**: Sample-exact theoretical foundation.
- 🛠️ Modular design: Drop into any C++ project or integrate as a JUCE module.

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
#include <cicuetea/CQT.h>

jsa::tea::CQT cqt;
cqt.prepare(sampleRate, fftSize, hopSize);

// Forward transform
auto spectrum = cqt.forward(inputBlock);

// Inverse transform
auto reconstructed = cqt.inverse(spectrum);
```

---

## 📀 Parameters & Design Notes

| Parameter       | Description                                  |
| --------------- | -------------------------------------------- |
| `binsPerOctave` | Controls resolution — typically 24 or 36     |
| `minFreq`       | Start of frequency range (e.g., 30 Hz)       |
| `maxFreq`       | Upper limit of transform                     |
| `gamma`         | Time/frequency scaling behavior              |
| `overlap`       | How much overlap exists in analysis windows  |
| `windowType`    | Type of analysis window (Gaussian preferred) |

> CiCueTea uses **nonzero Gaussian windows** for excellent frequency localization and smooth invertibility.

---

## 🧠 What Makes It Special?

Most CQT implementations either:

- Are not invertible,
- Are not usable in real time,
- Or compromise on time/frequency resolution.

**CiCueTea** is designed to achieve **all three**:

- Real-time forward/inverse streaming
- Almost perfect numerical reconstruction
- High resolution in perceptually-relevant bands

---

## 📦 Used in

- 🎛️ [`CiCuePitchScrambler`](https://github.com/your-org/CiCuePitchScrambler)
- 🎚️ [`CiCueEQ`](https://github.com/your-org/CiCueEQ)
- 🔊 [`CiCueSpectrum`](https://github.com/your-org/CiCueSpectrum)

---

## 🧳️ Name?

**CiCueTea** → "CQT"

A spectral engine so smooth, you’ll want a second cup.

---

## 🪖 License

MIT License — use it freely, sip responsibly.

---

## 👤 Author

Developed by [Juan Sierra](https://github.com/juansierradev) as part of research at NYU Abu Dhabi and Meyer Sound Labs.

---

## 🧪 Advanced Options

Want to explore field separation, harmonic freezing, or log-frequency spectral effects?

CiCueTea was designed for research-driven, next-generation DSP workflows. Reach out or contribute if you'd like to help expand it.

