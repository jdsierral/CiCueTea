## CQT library comparison

```
date:     2026-07-11
machine:  Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz (x86_64)
os:       macOS-10.16-x86_64-i386-64bit
python:   3.12.0
signal:   N = 2^20 = 1048576 samples @ 48000 Hz (21.8 s), best of 10 runs, both probes (noise + in-range sweep)
numpy:    1.26.4
scipy:    1.17.1
librosa:  0.11.0
nsgt:     0.18
torch:    2.2.2
nnAudio2: 2.0.2
cqt_pytorch:0.0.5
cicuetea: 41db0b8
```

### White noise — strict exactness probe (flat weighting, nothing in the band can hide)

| Implementation    | Configuration                                               | Round-trip RMS error | Coefficients (xN) | Forward (ms) | Inverse (ms) | x realtime |
|-------------------|-------------------------------------------------------------|----------------------|-------------------|--------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                               | 6.17e-16             | 8.49e+07 (81.00x) | 4479.3       | 3796.9       | 2.6x       |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                              | 6.41e-16             | 2.47e+06 (2.36x)  | 114.2        | 106.2        | 99x        |
| librosa           | 80 bins, 100-10000 Hz, default hop                          | 1.29e+00             | 1.64e+05 (0.16x)  | 75.8         | 121.6        | 111x       |
| nsgt (matrixform) | matrixform, 81 channels, 100-10000 Hz                       | 2.73e-15             | 4.95e+07 (47.25x) | 12461.0      | 12239.8      | 0.9x       |
| nsgt (ragged)     | ragged, 81 channels, 100-10000 Hz                           | 2.30e-15             | 1.36e+06 (1.29x)  | 297.7        | 304.6        | 36x        |
| cqt-pytorch       | 8 octaves, float32, torch 6 threads                         | 6.03e-02             | 5.82e+06 (5.55x)  | 94.3         | 92.0         | 117x       |
| nnAudio2          | CQT1992v2 + iCQT (Landweber, 32 iter), float32, default hop | 9.25e-01             | 1.64e+05 (0.16x)  | 60.8         | 11063.2      | 2.0x       |

### Log sweep 100-10000 Hz, Kaiser(beta=9) — realistic in-range streaming use case

| Implementation    | Configuration                                               | Round-trip RMS error | Coefficients (xN) | Forward (ms) | Inverse (ms) | x realtime |
|-------------------|-------------------------------------------------------------|----------------------|-------------------|--------------|--------------|------------|
| CiCueTea (dense)  | dense, 81 bands, 100-10000 Hz                               | 2.23e-16             | 8.49e+07 (81.00x) | 4863.9       | 3975.5       | 2.5x       |
| CiCueTea (sparse) | sparse, 81 bands, 100-10000 Hz                              | 2.17e-16             | 2.47e+06 (2.36x)  | 116.7        | 123.5        | 91x        |
| librosa           | 80 bins, 100-10000 Hz, default hop                          | 2.74e-01             | 1.64e+05 (0.16x)  | 78.5         | 144.5        | 98x        |
| nsgt (matrixform) | matrixform, 81 channels, 100-10000 Hz                       | 3.13e-16             | 4.95e+07 (47.25x) | 12807.4      | 12730.2      | 0.9x       |
| nsgt (ragged)     | ragged, 81 channels, 100-10000 Hz                           | 2.49e-16             | 1.36e+06 (1.29x)  | 315.0        | 313.1        | 35x        |
| cqt-pytorch       | 8 octaves, float32, torch 6 threads                         | 4.95e-06             | 5.82e+06 (5.55x)  | 102.4        | 104.9        | 105x       |
| nnAudio2          | CQT1992v2 + iCQT (Landweber, 32 iter), float32, default hop | 1.36e-01             | 1.64e+05 (0.16x)  | 39.7         | 8231.1       | 2.6x       |

