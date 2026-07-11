# Benchmarks

Quantitative backing for the README's "How It Compares" table: round-trip
reconstruction error and forward/inverse wall time, on the same task, for

* the **Python ecosystem** (`compare.py`): every invertible(-claiming) CQT
  implementation on PyPI vs the CiCueTea Python reference, and
* the **C++ engines** (`cpp/`): CiCueTea vs [The Gaborator](https://www.gaborator.com),
  its closest competitor (the only other invertible, streaming-capable C++ CQT).

No third-party code lives in this repository — Python libraries are fetched
from PyPI at reproduction time, and the Gaborator (AGPLv3-or-commercial) must
be downloaded from gaborator.com and pointed at explicitly (see below).

## Results

Latest run — Apple M2 Max, macOS, Python 3.12 (full provenance and archived runs
in [`results/`](results/)):

| Implementation    | Configuration                                            | Round-trip RMS error | Coefficients (xN) | Forward (ms) | Inverse (ms) | x realtime      |
|-------------------|----------------------------------------------------------|----------------------|-------------------|--------------|--------------|-----------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                            | 5.74e-16             | 8.49e+07 (81.00x) | 1916.7       | 1716.6       | 6.0x            |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                           | 5.98e-16             | 2.47e+06 (2.36x)  | 38.8         | 40.2         | 277x            |
| librosa           | 80 bins, 100-10000 Hz, default hop                       | 1.29e+00             | 1.64e+05 (0.16x)  | 34.0         | 68.9         | 212x            |
| nsgt (matrixform) | matrixform, 81 channels, 100-10000 Hz                    | 2.75e-15             | 4.95e+07 (47.25x) | 3913.5       | 4070.3       | 2.7x            |
| nsgt (ragged)     | ragged, 81 channels, 100-10000 Hz                        | 2.28e-15             | 1.36e+06 (1.29x)  | 122.0        | 114.6        | 92x             |
| cqt-pytorch       | 8 octaves, float32, torch 8 threads                      | 6.03e-02             | 5.82e+06 (5.55x)  | 152.9        | 156.6        | 71x             |
| nnAudio           | forward only (no exact inverse), float32, magnitude bins | n/a                  | 1.64e+05 (0.16x)  | 22.0         | n/a          | 993x (fwd only) |

How to read this:

* **Reconstruction error is the durable column** — it is machine-independent
  and measures whether the inverse actually reconstructs the signal. Only the
  NSGF-based implementations (CiCueTea, nsgt) achieve numerical-precision
  round trips. librosa's least-squares inverse loses more than the signal's
  own RMS on broadband input (its high bands are undersampled at the default
  hop); cqt-pytorch reconstructs to about -24 dB; nnAudio has no exact inverse.
* **Coefficients (xN)** is the representation's redundancy: stored coefficient
  values (complex, except nnAudio's magnitudes) relative to the N input
  samples. Exact reconstruction is only meaningful together with this number —
  anyone can be invertible at 81x oversampling (dense/matrixform rasterized
  forms, meant for analysis/display, not storage). The telling cases: librosa
  sits at **0.16x — below critical sampling, which is *why* its high bands
  cannot reconstruct**; ragged nsgt is the most compact exact frame (1.29x);
  CiCueTea sparse pays a modest premium over it (2.36x, spans padded to
  powers of two — that padding is where the ~2.6x speed advantage comes from).
* **Wall times are machine-specific.** The relative ordering is fairly stable;
  the absolute milliseconds are not. Among the exact-reconstruction
  implementations, CiCueTea sparse is ~2.6x faster than ragged nsgt and ~90x
  faster than matrixform nsgt on this machine.
* These numbers time the **Python reference implementation**. The C++ engine
  is the production path: its performance is exercised by the bench-labeled
  test groups (`ctest -L bench`), and its real-time claim (no allocation on
  the processing path) is *enforced* in the test suite via Eigen's runtime
  malloc checks, not benchmarked here.

## C++: CiCueTea vs The Gaborator and rt-cqt

Same task, compiled Release, every library on its fastest available FFT
backend (vDSP for CiCueTea and the Gaborator on macOS; rt-cqt's pffft is
scalar by its own design, `SIMD_SZ 1`). Latest run — Apple M2 Max, clang 21,
Eigen 3.5 (archived in [`results/`](results/)):

| Implementation    | Configuration                                             | Round-trip RMS error | Coefficients (xN) | Forward (ms)   | Inverse (ms) | x realtime |
|-------------------|------------------------------------------------------------|----------------------|-------------------|----------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                             | 6.03e-16             | 8.49e+07 (81.00x) | 831.5          | 1120.8       | 11x        |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                            | 6.08e-16             | 2.47e+06 (2.36x)  | 28.3           | 28.2         | 387x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double      | 7.88e-08             | 5.02e+06 (4.79x)  | 37.5           | 68.4         | 206x       |
| rt-cqt            | 96 bins, 8 octaves below Nyquist, hop 256, streamed 1024  | 1.34e+00             | n/a (TBD)         | 99.6 (fwd+inv) | n/a          | 219x       |

Against its closest competitor, CiCueTea sparse wins on all three axes: eight
orders of magnitude closer to exact reconstruction, ~2x faster, and half the
coefficient footprint (2.36x vs 4.79x redundancy — the Gaborator's count
includes coefficients extending past the signal edges by the atoms' time
support, which is its true storage cost).

**Gaborator** — fairness notes, both cutting *against* CiCueTea: its scale
always extends to Nyquist, so it computes 97 bands where CiCueTea is asked for
100–10000 Hz (81 bands); and its coefficient container is allocated inside the
timed forward pass because allocation during analysis is inherent to its API —
which is also exactly why it cannot run inside a real-time audio callback
(CiCueTea's no-allocation processing path is enforced by the test suite via
Eigen's runtime malloc checks). Its ~1e-7 reconstruction error is its
documented design point, not a bug.

**rt-cqt** — run in its designed streaming mode (1024-sample blocks, fresh
instance per repetition since its internal buffers don't support re-runs, and
forward/inverse interleaved by its hop schedule, so only the combined time is
measurable). The error needs explanation: the output comes back at full energy
but ~1.34 RMS (≈ √2, i.e. uncorrelated) from the input *at the best possible
global alignment*, found by circular cross-correlation. This is not noise —
rt-cqt's multirate pipeline gives each octave a different latency (the lowest
octave's hop is 128x the highest's), so the reconstruction is dispersed
per-band and no time shift can recover the waveform. Individual bands sound
fine; the broadband waveform never re-assembles. That is the measurable
meaning of "kernel-based, not formally invertible" in the main README.

To reproduce:

```sh
# Gaborator: download from gaborator.com (AGPL — not vendored, see below)
# rt-cqt:    git clone --recursive https://github.com/jmerkt/rt-cqt (BSD-3)
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DGABORATOR_DIR=/path/to/gaborator-2.1 \
      -DRTCQT_DIR=/path/to/rt-cqt
cmake --build cpp/build -j
./cpp/build/cqt_bench > results/$(date +%F)-<machine>-cpp.md
```

Either `*_DIR` may be omitted; the CiCueTea engines are always benchmarked.

## Methodology

* Task: analyze and resynthesize 21.8 s of white noise (2^20 samples at
  48 kHz), 12 bands/octave over 100–10000 Hz where the library's interface
  allows that range to be specified (each row's actual configuration is in the
  table). White noise is the adversarial case for invertibility: it has energy
  everywhere, so any under-covered frequency region shows up in the error.
* Timing: `time.perf_counter` around forward and inverse separately, best of
  3 runs (absorbs one-time costs such as librosa's numba JIT on first call).
  Error is RMS of `x - inverse(forward(x))`.
* Coefficient counts are stored values as the library hands them back
  (complex counts as one). The (xN) redundancy factor is count / N input
  samples; a complete, stably invertible frame needs at least 1x
  (critical sampling) — anything below cannot reconstruct broadband input.
* Fairness caveats: the libraries do not compute identical objects — librosa
  and nnAudio produce hop-decimated frames, the NSGF implementations produce
  critically-sampled (sparse/ragged) or fully-rasterized (dense/matrixform)
  coefficients, and the torch-based libraries run in float32. The comparison
  is at the *task* level (same signal in, resynthesized signal out), which is
  the level a user experiences.
* Device policy: **everything runs on the CPU**, deliberately. The comparison
  is algorithmic (same silicon for every row), and the target use case —
  real-time audio, with a per-block budget of a few milliseconds — rules out
  GPU dispatch regardless (the host-device transfer round trip alone breaks a
  callback budget). nnAudio and cqt-pytorch are designed for GPU *batch/ML*
  pipelines and would post much higher throughput there; that is a different
  benchmark for a different use case. Note the asymmetry that remains runs in
  their favor: torch parallelizes across all cores (thread count in the
  config column) while CiCueTea's numbers are single-threaded.

## Reproducing

```sh
pip install -r requirements.txt
python compare.py            # options: --repeats N --samples M
```

Every run prints its own environment report (machine, OS, Python, library
versions, CiCueTea commit), so pasted output is self-documenting. To archive a
run, redirect stdout to `results/<date>-<machine>.md`. Runs with missing
libraries still work — absent entries are skipped and listed.

A note on **nsgt**: the project is dormant. Its 2017 PyPI release no longer
builds on modern Python, and git master crashes with numpy >= 2 (an integer
array clipped against float `inf` in `nsgfwin_sl.py`). `requirements.txt`
therefore pins the commit from the open upstream fix
([grrrr/nsgt#38](https://github.com/grrrr/nsgt/pull/38)) — verified to
reproduce the archived numbers exactly. Use Python <= 3.13 (its `setup.py`
does not build on 3.14).

**The Gaborator** is deliberately excluded from the runnable set: it is
AGPLv3-or-commercial, so this MIT-licensed repository neither vendors nor
depends on it. To compare against it, download it from
[gaborator.com](https://www.gaborator.com) and run its `streaming` example on
the same signal.
