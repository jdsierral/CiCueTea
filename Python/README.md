# cicuetea

Python reference implementation of **CiCueTea**, a real-time, invertible
Constant-Q Transform (CQT) engine based on nonstationary Gabor frames (NSGF).

This package is an offline, NumPy/SciPy reference — dense and sparse forward/inverse
transforms, sparse-to-dense rasterization, and the block slicing/splicing utilities used to
build a streaming pipeline on top of it. The real-time, allocation-free implementation lives
in the C++ library this package accompanies.

```python
import numpy as np
from cicuetea import NsgfCQT

fs, n_samples, frac, f_min, f_max = 48000, 2**16, 1 / 48, 100, 10000
x = np.random.randn(n_samples)

cqt = NsgfCQT("dense", fs, n_samples, frac, f_min, f_max)
X = cqt.forward(x)
y = cqt.inverse(X)
```

Full documentation, the C++ engine, and the MATLAB reference implementation live at
<https://github.com/jdsierral/CiCueTea>.

## License

MIT — see [LICENSE](https://github.com/jdsierral/CiCueTea/blob/main/LICENSE).
