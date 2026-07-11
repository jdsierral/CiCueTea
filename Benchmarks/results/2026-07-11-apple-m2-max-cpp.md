## CQT C++ benchmark

```
date:     2026-07-11
machine:  Apple M2 Max
compiler: AppleClang 21.0.0.21000101
eigen:    3.5.0
gaborator: 2.1
signal:   N = 2^20 = 1048576 samples @ 48000 Hz (21.8453 s), best of 10 runs, both probes (noise + in-range sweep)
```

### White noise - strict exactness probe (flat weighting, nothing in the band can hide)

| Implementation    | Configuration                                                                                 | Round-trip RMS error | Coefficients (xN) | Forward (ms)   | Inverse (ms) | x realtime |
|-------------------|-----------------------------------------------------------------------------------------------|----------------------|-------------------|----------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                                                                 | 6.03e-16             | 8.49e+07 (81.00x) | 810.2          | 1099.2       | 11x        |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                                                                | 6.08e-16             | 2.47e+06 (2.36x)  | 25.9           | 25.7         | 423x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double                                          | 7.88e-08             | 5.02e+06 (4.79x)  | 37.4           | 69.7         | 204x       |
| rt-cqt            | 96 bins, 8 octaves below Nyquist, hop 256, streamed 1024-sample blocks, best global alignment | 1.34e+00             | n/a               | 99.4 (fwd+inv) | n/a          | 220x       |

### Log sweep 100-10000 Hz, Kaiser(beta=9) - realistic in-range streaming use case

| Implementation    | Configuration                                                                                 | Round-trip RMS error | Coefficients (xN) | Forward (ms)    | Inverse (ms) | x realtime |
|-------------------|-----------------------------------------------------------------------------------------------|----------------------|-------------------|-----------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                                                                 | 2.25e-16             | 8.49e+07 (81.00x) | 798.1           | 1092.2       | 12x        |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                                                                | 2.17e-16             | 2.47e+06 (2.36x)  | 25.9            | 25.2         | 427x       |
| Gaborator         | 12 bpo, 97 bands, 100 Hz-Nyquist (by design), double                                          | 5.13e-08             | 5.02e+06 (4.79x)  | 39.7            | 71.6         | 196x       |
| rt-cqt            | 96 bins, 8 octaves below Nyquist, hop 256, streamed 1024-sample blocks, best global alignment | 3.88e-01             | n/a               | 105.8 (fwd+inv) | n/a          | 206x       |
