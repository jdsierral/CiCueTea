## CQT C++ benchmark

```
date:     2026-07-11
machine:  Apple M2 Max
compiler: clang 21.0.0 (clang-2100.1.1.101)
eigen:    3.5.0
gaborator: 2.1
signal:   white noise, N = 2^20 = 1048576 samples @ 48000 Hz (21.8453 s), best of 3 runs
```

| Implementation    | Configuration                                                                                 | Round-trip RMS error | Coefficients (xN) | Forward (ms)   | Inverse (ms) | x realtime |
|-------------------|-----------------------------------------------------------------------------------------------|----------------------|-------------------|----------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                                                                 | 6.03e-16             | 8.49e+07 (81.00x) | 831.5          | 1120.8       | 11x        |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                                                                | 6.08e-16             | 2.47e+06 (2.36x)  | 28.3           | 28.2         | 387x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double                                          | 7.88e-08             | 5.02e+06 (4.79x)  | 37.5           | 68.4         | 206x       |
| rt-cqt            | 96 bins, 8 octaves below Nyquist, hop 256, streamed 1024-sample blocks, best global alignment | 1.34e+00             | n/a               | 99.6 (fwd+inv) | n/a          | 219x       |
