## CQT C++ benchmark

```
date:     2026-07-12
machine:  Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz
compiler: AppleClang 16.0.0.16000026
eigen:    3.5.0
gaborator: 2.1
rt-cqt:   not built (set -DRTCQT_DIR to include it)
signal:   N = 2^20 = 1048576 samples @ 48000 Hz (21.8453 s), best of 10 runs, both probes (noise + in-range sweep)
```

### White noise - strict exactness probe (flat weighting, nothing in the band can hide)

| Implementation    | Configuration                                        | Round-trip RMS error | Coefficients (xN) | Forward (ms) | Inverse (ms) | x realtime |
|-------------------|------------------------------------------------------|----------------------|-------------------|--------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                        | 7.32e-16             | 8.49e+07 (81.00x) | 1863.0       | 2460.5       | 5.1x       |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                       | 7.23e-16             | 2.47e+06 (2.36x)  | 49.9         | 46.6         | 226x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double | 7.88e-08             | 5.02e+06 (4.79x)  | 85.8         | 124.3        | 104x       |

### Log sweep 100-10000 Hz, Kaiser(beta=9) - realistic in-range streaming use case

| Implementation    | Configuration                                        | Round-trip RMS error | Coefficients (xN) | Forward (ms) | Inverse (ms) | x realtime |
|-------------------|------------------------------------------------------|----------------------|-------------------|--------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                        | 2.69e-16             | 8.49e+07 (81.00x) | 1824.3       | 2418.6       | 5.1x       |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                       | 2.57e-16             | 2.47e+06 (2.36x)  | 46.8         | 44.4         | 239x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double | 5.13e-08             | 5.02e+06 (4.79x)  | 88.8         | 109.0        | 110x       |
