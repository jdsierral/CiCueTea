# Benchmarks

Quantitative backing for the README's "How It Compares" table: round-trip
reconstruction error and forward/inverse wall time for the invertible(-claiming)
CQT implementations available on PyPI, measured against the **CiCueTea Python
reference** on the same task.

No third-party code lives in this repository — every library is fetched from
PyPI at reproduction time (see *Reproducing* below).

## Results

Latest run — Apple M2 Max, macOS, Python 3.12 (full provenance and archived runs
in [`results/`](results/)):

| Implementation    | Configuration                            | Round-trip RMS error | Forward (ms) | Inverse (ms) | x realtime      |
|-------------------|------------------------------------------|----------------------|--------------|--------------|-----------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz            | 5.74e-16             | 1858.6       | 1691.2       | 6.2x            |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz           | 5.98e-16             | 44.4         | 42.4         | 252x            |
| librosa           | 80 bins, 100-10000 Hz, default hop       | 1.29e+00             | 28.8         | 58.9         | 249x            |
| nsgt (matrixform) | matrixform, 81 channels, 100-10000 Hz    | 2.75e-15             | 3749.1       | 4080.7       | 2.8x            |
| nsgt (ragged)     | ragged, 81 channels, 100-10000 Hz        | 2.28e-15             | 118.6        | 111.9        | 95x             |
| cqt-pytorch       | 8 octaves, float32, torch 8 threads      | 6.03e-02             | 155.4        | 155.0        | 70x             |
| nnAudio           | forward only (no exact inverse), float32 | n/a                  | 22.8         | n/a          | 960x (fwd only) |

How to read this:

* **Reconstruction error is the durable column** — it is machine-independent
  and measures whether the inverse actually reconstructs the signal. Only the
  NSGF-based implementations (CiCueTea, nsgt) achieve numerical-precision
  round trips. librosa's least-squares inverse loses more than the signal's
  own RMS on broadband input (its high bands are undersampled at the default
  hop); cqt-pytorch reconstructs to about -24 dB; nnAudio has no exact inverse.
* **Wall times are machine-specific.** The relative ordering is fairly stable;
  the absolute milliseconds are not. Among the exact-reconstruction
  implementations, CiCueTea sparse is ~2.6x faster than ragged nsgt and ~90x
  faster than matrixform nsgt on this machine.
* These numbers time the **Python reference implementation**. The C++ engine
  is the production path: its performance is exercised by the bench-labeled
  test groups (`ctest -L bench`), and its real-time claim (no allocation on
  the processing path) is *enforced* in the test suite via Eigen's runtime
  malloc checks, not benchmarked here.

## Methodology

* Task: analyze and resynthesize 21.8 s of white noise (2^20 samples at
  48 kHz), 12 bands/octave over 100–10000 Hz where the library's interface
  allows that range to be specified (each row's actual configuration is in the
  table). White noise is the adversarial case for invertibility: it has energy
  everywhere, so any under-covered frequency region shows up in the error.
* Timing: `time.perf_counter` around forward and inverse separately, best of
  3 runs (absorbs one-time costs such as librosa's numba JIT on first call).
  Error is RMS of `x - inverse(forward(x))`.
* Fairness caveats: the libraries do not compute identical objects — librosa
  and nnAudio produce hop-decimated frames, the NSGF implementations produce
  critically-sampled (sparse/ragged) or fully-rasterized (dense/matrixform)
  coefficients, and the torch-based libraries run in float32. The comparison
  is at the *task* level (same signal in, resynthesized signal out), which is
  the level a user experiences.

## Reproducing

```sh
pip install -r requirements.txt
python compare.py            # options: --repeats N --samples M
```

Every run prints its own environment report (machine, OS, Python, library
versions, CiCueTea commit), so pasted output is self-documenting. To archive a
run, redirect stdout to `results/<date>-<machine>.md`. Runs with missing
libraries still work — absent entries are skipped and listed.

**The Gaborator** is deliberately excluded from the runnable set: it is
AGPLv3-or-commercial, so this MIT-licensed repository neither vendors nor
depends on it. To compare against it, download it from
[gaborator.com](https://www.gaborator.com) and run its `streaming` example on
the same signal.
