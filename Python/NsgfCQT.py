import numpy as np
import scipy.fft as fft


class NsgfCQT:
    def __init__(self, mode, sample_rate, n_samples, 
                 frac=1/12, f_min=1e2, f_max=1e4, f_ref=1e3, threshold=1e-6):
        sample_rate = float(sample_rate)
        n_samples = int(n_samples)
        frac = float(frac)
        f_min = float(f_min)
        f_max = float(f_max)
        f_ref = float(f_ref)
        
        # Structural validity
        assert sample_rate > 0, "Sample rate must be positive"
        assert n_samples > 0, "Block must be non-empty"
        if mode == "sparse":  # power-of-two spans (pad_indices) assume it
            assert (n_samples & (n_samples - 1)) == 0, "n_samples must be a power of 2"
        assert frac > 0, "frac (reciprocal of bands per octave) must be positive"
        assert f_ref > 0, "Reference frequency must be positive"
        assert f_min > 0, "Lower bound must be positive"
        assert f_min < f_max, "The range must be a range (f_min < f_max)"
        assert f_max * 2 < sample_rate, "Higher bound must be below nyquist"

        self.min_bw = 4

        n_bands_up = int(np.ceil(1 / frac * np.log2(f_max / f_ref)))
        n_bands_dn = int(np.ceil(1 / frac * np.log2(f_ref / f_min)))
        n_bands = n_bands_dn + n_bands_up + 1
        n_freqs = n_samples
        bands = np.arange(-n_bands_dn, n_bands_up + 1)
        band_axis = f_ref * 2.0 ** (frac * bands)               
        time_axis = np.arange(-n_samples / 2, n_samples / 2) / sample_rate
        freq_axis = np.arange(n_freqs) * sample_rate / n_freqs  

        c = np.log(4) / (frac ** 2.0)                   # Horizontal Scale Factor
        # log2(0) at the DC bin is fine: -inf yields a Gaussian weight of exactly 0
        with np.errstate(divide="ignore"):
            outer_diff = np.subtract.outer(np.log2(freq_axis), np.log2(band_axis))
        g = np.exp(-c * outer_diff ** 2.0)              # Analytic Gaussians
        g[np.where(freq_axis < band_axis[0]), 0] = 1    # Make lowest band an LPF
        g[np.where(freq_axis > band_axis[-1]), -1] = 1  # Make highest band an HPF

        if mode == "sparse":                # In sparse mode truncate gaussians
            g[np.where(g < threshold)] = 0

        d = np.sum(g ** 2.0, 1)

        # Measured frame health. Two failure modes: coverage gaps (bins no atom reaches) show up as an ill-conditioned 
        # frame operator; unresolved atoms (bands too narrow for the grid) are invisible in d and are caught by per-atom 
        # support instead. (The old identity check rms(sum(g*g_dual) - 1) was algebraically zero by construction — it 
        # could only ever catch exact 0/0 NaNs.)
        dh = d[freq_axis <= sample_rate / 2]
        assert dh.min() > 1e-6 * dh.max(), \
            "Frame operator ill-conditioned: coverage gaps (Q too high or threshold too aggressive)"
        support = np.count_nonzero(g > threshold, axis=0)
        assert support.min() >= self.min_bw, \
            "Atoms unresolved by the frequency grid (Q too high for this block size)"

        g_dual = g / d[:, None]             # Compute the dual frame

        g[np.where(freq_axis > sample_rate / 2), :] = 0
        g_dual[np.where(freq_axis > sample_rate / 2), :] = 0

        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_freqs = n_freqs
        self.n_bands = n_bands
        self.f_min = f_min
        self.f_max = f_max
        self.f_ref = f_ref
        self.time_axis = time_axis
        self.band_axis = band_axis
        self.freq_axis = freq_axis
        self.g = g
        self.d = d
        self.g_dual = g_dual
        self.mode = mode

        if mode == "sparse":
            idxs = [None] * n_bands
            g_list = [None] * n_bands
            g_dual_list = [None] * n_bands
            shifts = [None] * n_bands

            for k in range(n_bands):
                ii = np.where(g[:, k] != 0)[0]
                ii = NsgfCQT.pad_indices(ii)
                n_coefs = len(ii)
                offset = ii[0]
                idxs[k] = ii
                n = np.arange(n_coefs)
                shifts[k] = np.exp(1j * 2 * np.pi * offset * n / n_coefs)
                g_list[k] = g[ii,k]
                g_dual_list[k] = g_dual[ii,k]

            self.g = g_list
            self.g_dual = g_dual_list
            self.idxs = idxs
            self.shifts = shifts


    def forward(self, x):
        X = fft.fft(x, self.n_samples) / self.n_samples

        if self.mode == "dense":
            X_cq = 2 * self.n_samples * fft.ifft(X[:, None] * self.g, self.n_samples, 0)
        elif self.mode == "sparse":
            X_cq = [None] * self.n_bands
            for k in range(self.n_bands):
                n_coefs = float(len(self.idxs[k]))
                Xi = fft.ifft(X[self.idxs[k]] * self.g[k])
                Xi = Xi * 2 * n_coefs * self.shifts[k]
                X_cq[k] = Xi
        return X_cq


    def inverse(self, X_cq):
        if self.mode == "dense":
            X = 1.0 / (2.0 * self.n_samples) * np.sum(fft.fft(X_cq, self.n_samples, 0) * self.g_dual, 1)
        elif self.mode == "sparse":
            X = np.zeros(self.n_samples, dtype=np.complex128)
            for k in range(self.n_bands):
                n_coefs = len(self.idxs[k])
                Xi = X_cq[k] * (self.shifts[k].conj() / (2.0 * n_coefs))
                Xi = fft.fft(Xi) * self.g_dual[k]
                X[self.idxs[k]] += Xi
        x = fft.irfft(X, self.n_samples) * self.n_samples
        return x
    

    @staticmethod
    def pad_indices(indices):
        i0 = indices[0]
        n_idx = indices.size
        if n_idx < 4:
            n_idx = 4
        n_idx_pow2 = int(2 ** np.ceil(np.log2(n_idx)))
        indices = i0 + np.arange(n_idx_pow2)
        return indices
    

    def rasterize(self, X_cq):
        """Reconstruct the dense [n_samples, n_bands] representation from
        sparse coefficients: undo the per-band phase and scaling, recover the
        band's span spectrum, and embed it at its true bins. Exactly equal to
        the dense transform's columns (up to the sparsity threshold), unlike
        naive bandlimited interpolation, which assumes baseband input while
        the coefficients carry the (aliased) carrier."""
        X_r = np.zeros([self.n_samples, self.n_bands], dtype=np.complex128)
        for k in range(self.n_bands):
            n_coefs = len(self.idxs[k])
            Xi = X_cq[k] * (self.shifts[k].conj() / (2.0 * n_coefs))
            spec = np.zeros(self.n_freqs, dtype=np.complex128)
            spec[self.idxs[k]] = fft.fft(Xi)
            X_r[:, k] = 2 * self.n_samples * fft.ifft(spec)
        return X_r
    
class NsgfVQT(NsgfCQT):
    def __init__(self, mode, sample_rate, n_samples, f_map=np.log2,
                 frac=1/12, f_min=1e2, f_max=1e4, f_ref=1e3, threshold=1e-6):
        sample_rate = float(sample_rate)
        n_samples = int(n_samples)
        frac = float(frac)
        f_min = float(f_min)
        f_max = float(f_max)
        f_ref = float(f_ref)
        # Structural validity: each condition provably breaks the transform on
        # its own. Feasibility with fuzzy boundaries (Q vs. block length) is
        # measured on the constructed frame below, not predicted from the
        # parameters; sub-octave ranges are legitimate. (The C++ version
        # additionally requires power-of-two n_samples in both modes, since
        # its FFT backends do.)
        assert sample_rate > 0, "Sample rate must be positive"
        assert n_samples > 0, "Block must be non-empty"
        if mode == "sparse":  # power-of-two spans (pad_indices) assume it
            assert (n_samples & (n_samples - 1)) == 0, "n_samples must be a power of 2"
        assert frac > 0, "frac (reciprocal of bands per octave) must be positive"
        assert f_ref > 0, "Reference frequency must be positive"
        assert f_min > 0, "Lower bound must be positive"
        assert f_min < f_max, "The range must be a range (f_min < f_max)"
        assert f_max * 2 < sample_rate, "Higher bound must be below nyquist"

        self.min_bw = 4

        n_bands_up = int(np.ceil(1 / frac * (f_map(f_max) - f_map(f_ref))))
        n_bands_dn = int(np.ceil(1 / frac * (f_map(f_ref) - f_map(f_min))))
        n_bands = n_bands_dn + n_bands_up + 1
        n_freqs = n_samples
        bands = np.arange(-n_bands_dn, n_bands_up + 1)
        band_axis = f_map(f_ref) + (frac * bands)               
        time_axis = np.arange(-n_samples / 2, n_samples / 2) / sample_rate 
        freq_axis = np.arange(n_freqs) * sample_rate / n_freqs  

        c = np.log(4) / (frac ** 2.0)                   # Horizontal Scale Factor
        # f_map(0) at the DC bin may be -inf (e.g. log2): Gaussian weight becomes exactly 0
        with np.errstate(divide="ignore"):
            warped_fax = f_map(freq_axis)
        outer_diff = np.subtract.outer(warped_fax, band_axis)
        g = np.exp(-c * outer_diff ** 2.0)              # Analytic Gaussians
        g[np.where(warped_fax < band_axis[0]), 0] = 1    # Make lowest band an LPF
        g[np.where(warped_fax > band_axis[-1]), -1] = 1  # Make highest band an HPF

        if mode == "sparse":                # In sparse mode truncate gaussians
            g[np.where(g < threshold)] = 0

        d = np.sum(g ** 2.0, 1)

        # Measured frame health (mirrors the C++ checks). Two failure modes:
        # coverage gaps (bins no atom reaches) show up as an ill-conditioned
        # frame operator; unresolved atoms (bands too narrow for the grid)
        # are invisible in d and are caught by per-atom support instead.
        # (The old identity check rms(sum(g*g_dual) - 1) was algebraically
        # zero by construction — it could only ever catch exact 0/0 NaNs.)
        dh = d[freq_axis <= sample_rate / 2]
        assert dh.min() > 1e-6 * dh.max(), \
            "Frame operator ill-conditioned: coverage gaps (Q too high or threshold too aggressive)"
        support = np.count_nonzero(g > threshold, axis=0)
        assert support.min() >= self.min_bw, \
            "Atoms unresolved by the frequency grid (Q too high for this block size)"

        g_dual = g / d[:, None]             # Compute the dual frame

        g[np.where(freq_axis > sample_rate / 2), :] = 0
        g_dual[np.where(freq_axis > sample_rate / 2), :] = 0

        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_freqs = n_freqs
        self.n_bands = n_bands
        self.f_map = f_map
        self.f_min = f_min
        self.f_max = f_max
        self.f_ref = f_ref
        self.time_axis = time_axis
        self.band_axis = band_axis
        self.freq_axis = freq_axis
        self.g = g
        self.d = d
        self.g_dual = g_dual
        self.mode = mode

        if mode == "sparse":
            idxs = [None] * n_bands
            g_list = [None] * n_bands
            g_dual_list = [None] * n_bands
            shifts = [None] * n_bands

            for k in range(n_bands):
                ii = np.where(g[:, k] != 0)[0]
                ii = NsgfCQT.pad_indices(ii)
                n_coefs = len(ii)
                offset = ii[0]
                idxs[k] = ii
                n = np.arange(n_coefs)
                shifts[k] = np.exp(1j * 2 * np.pi * offset * n / n_coefs)
                g_list[k] = g[ii,k]
                g_dual_list[k] = g_dual[ii,k]

            self.g = g_list
            self.g_dual = g_dual_list
            self.idxs = idxs
            self.shifts = shifts

