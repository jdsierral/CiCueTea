import numpy as np
import scipy.fft as fft
import scipy.signal as sg


class NsgfCQT:
    def __init__(self, mode, sample_rate, n_samples, 
                 frac=1/12, f_min=1e2, f_max=1e4, f_ref=1e3, threshold=1e-6):
        sample_rate = float(sample_rate)
        n_samples = int(n_samples)
        frac = float(frac)
        f_min = float(f_min)
        f_max = float(f_max)
        f_ref = float(f_ref)
        assert f_min != 0, "Lower bound can't be 0"
        assert f_max * 2 < sample_rate, "Higher bound must be below nyquist"
        assert f_min * 2 <= f_max, "Leave at least 1 octave"
        # assert sample_rate / (f_min * ((2.0**frac) - 1.0)) > n_samples, "Q is too high"

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
        outer_diff = np.subtract.outer(np.log2(freq_axis), np.log2(band_axis))
        g = np.exp(-c * outer_diff ** 2.0)              # Analytic Gaussians
        g[np.where(freq_axis < band_axis[0]), 0] = 1    # Make lowest band an LPF
        g[np.where(freq_axis > band_axis[-1]), -1] = 1  # Make highest band an HPF

        if mode == "sparse":                # In sparse mode truncate gaussians
            g[np.where(g < threshold)] = 0

        d = np.sum(g ** 2.0, 1)
        g_dual = g / d[:, None]             # Compute the dual frame

        # Check for invertibility
        assert np.sqrt(np.mean((np.sum(g * g_dual, 1) - 1) ** 2.0)) < 1e-10

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

        if self.mode == "full":
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
        if self.mode == "full":
            X = 1.0 / (2.0 * self.n_samples) * np.sum(fft.fft(X_cq, self.n_samples, 0) * self.g_dual, 1)
        elif self.mode == "sparse":
            X = np.zeros(self.n_samples, dtype=np.complex128)
            for k in range(self.n_bands):
                n_coefs = len(self.idxs[k])
                Xi = X_cq[k]
                Xi *= self.shifts[k].conj() / (2.0 * n_coefs)
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
        X_r = np.zeros([self.n_samples, self.n_bands], dtype=np.complex128)
        for k in range(self.n_bands):
            X_r[:,k] = sg.resample(X_cq[k], self.n_samples)
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
        assert f_min != 0, "Lower bound can't be 0"
        assert f_max * 2 < sample_rate, "Higher bound must be below nyquist"
        assert f_min * 2 <= f_max, "Leave at least 1 octave"
        # assert sample_rate / (f_min * ((2.0**frac) - 1.0)) > n_samples, "Q is too high"

        self.min_bw = 4

        n_bands_up = int(np.ceil(1 / frac * (f_map(f_max) - f_map(f_min))))
        n_bands_dn = int(np.ceil(1 / frac * (f_map(f_ref) - f_map(f_min))))
        n_bands = n_bands_dn + n_bands_up + 1
        n_freqs = n_samples
        bands = np.arange(-n_bands_dn, n_bands_up + 1)
        band_axis = f_map(f_ref) + (frac * bands)               
        time_axis = np.arange(-n_samples / 2, n_samples / 2) / sample_rate 
        freq_axis = np.arange(n_freqs) * sample_rate / n_freqs  

        c = np.log(4) / (frac ** 2.0)                   # Horizontal Scale Factor
        outer_diff = np.subtract.outer(f_map(freq_axis), band_axis)
        g = np.exp(-c * outer_diff ** 2.0)              # Analytic Gaussians
        g[np.where(f_map(freq_axis) < band_axis[0]), 0] = 1    # Make lowest band an LPF
        g[np.where(f_map(freq_axis) > band_axis[-1]), -1] = 1  # Make highest band an HPF

        if mode == "sparse":                # In sparse mode truncate gaussians
            g[np.where(g < threshold)] = 0

        d = np.sum(g ** 2.0, 1)
        g_dual = g / d[:, None]             # Compute the dual frame

        # Check for invertibility
        assert np.sqrt(np.mean((np.sum(g * g_dual, 1) - 1) ** 2.0)) < 1e-10

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

