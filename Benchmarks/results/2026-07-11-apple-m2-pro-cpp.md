## CQT C++ benchmark

```
date:     2026-07-11
machine:  Apple M2 Pro
compiler: AppleClang 21.0.0.21000101
eigen:    3.4.0
gaborator: 2.1
signal:   N = 2^20 = 1048576 samples @ 48000 Hz (21.8453 s), best of 10 runs, both probes (noise + in-range sweep)
```

### White noise - strict exactness probe (flat weighting, nothing in the band can hide)

| Implementation    | Configuration                                                                                 | Round-trip RMS error | Coefficients (xN) | Forward (ms)   | Inverse (ms) | x realtime |
|-------------------|-----------------------------------------------------------------------------------------------|----------------------|-------------------|----------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                                                                 | 6.03e-16             | 8.49e+07 (81.00x) | 784.0          | 1059.8       | 12x        |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                                                                | 6.09e-16             | 2.47e+06 (2.36x)  | 25.3           | 24.6         | 438x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double                                          | 7.88e-08             | 5.02e+06 (4.79x)  | 37.5           | 69.0         | 205x       |
| rt-cqt            | 96 bins, 8 octaves below Nyquist, hop 256, streamed 1024-sample blocks, best global alignment | 1.34e+00             | n/a               | 96.7 (fwd+inv) | n/a          | 226x       |

### Log sweep 100-10000 Hz, Kaiser(beta=9) - realistic in-range streaming use case

| Implementation    | Configuration                                                                                 | Round-trip RMS error | Coefficients (xN) | Forward (ms)   | Inverse (ms) | x realtime |
|-------------------|-----------------------------------------------------------------------------------------------|----------------------|-------------------|----------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                                                                 | 2.25e-16             | 8.49e+07 (81.00x) | 784.6          | 1065.4       | 12x        |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                                                                | 2.17e-16             | 2.47e+06 (2.36x)  | 25.8           | 25.2         | 428x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double                                          | 5.13e-08             | 5.02e+06 (4.79x)  | 37.5           | 68.9         | 205x       |
| rt-cqt            | 96 bins, 8 octaves below Nyquist, hop 256, streamed 1024-sample blocks, best global alignment | 3.88e-01             | n/a               | 97.0 (fwd+inv) | n/a          | 225x       |
