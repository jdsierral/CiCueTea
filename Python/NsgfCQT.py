import numpy as np
import scipy.fft as fft
import scipy.signal as sg

class NsgfCQT:
    def __init__(self, type, fs, nSamps, frac=1/12, fMin=1e2, fMax=1e4, fRef=1e3, th=1e-6):

        assert(fMin != 0, "Lower bound cant be 0")
        assert(fMax * 2 < fs, "Higher bound must be below nyquist")
        assert(fMin * 2 <= fMax, "Leave at least 1 octave")
        assert(fs / (fMin * (2.0**(frac) - 1)) > nSamps, "Q is too high")

        self.minBW = 4

        nBandsUp = int(np.ceil(1/frac * np.log2(fMax / fRef)))
        nBandsDn = int(np.ceil(1/frac * np.log2(fRef / fMin)))
        nBands = nBandsDn + nBandsUp + 1
        nFreqs = nSamps
        bax = fRef * 2.0**(frac * np.arange(-nBandsDn, nBandsUp))   # Band Axis
        tax = np.arange(-nSamps/2, nSamps/2)                        # Time Axis
        fax = np.arange(nFreqs) * fs / nFreqs                       # Frequency Axis

        c = np.log(4) / (frac**2.0)                                 # Horizontal Scale Factor
        outerDif = np.subtract.outer(np.log2(bax), np.log2(fax))
        g = np.exp(-c * outerDif**2.0)                              # Analytic Gaussains
        g[ 0, np.where(fax < bax[ 0])] = 1                          # Make lowest band an LPF
        g[-1, np.where(fax > bax[-1])] = 1                          # Make highest band an HPF

        if type == "sparse":                                        # In sparse mode truncate
            g[np.where(g < th)] = 0                                 # gaussians

        d = np.sum(g**2.0,0)                    
        gDual = g / d                                               # Compute the dual frame

        assert(np.sqrt(np.mean((np.sum(g * gDual, 2) - 1)**2.0)) < 1e-10)
                                                                    # Check for invertibility

        g[:,np.where(fax > fs/2)] = 0                               # Zero after nyquist to
        gDual[:,np.where(fax > fs/2)] = 0                           # avoid conjugate calculation

        self.fs = fs                                                # store all the data
        self.nSamps = nSamps
        self.nFreqs = nFreqs
        self.nBands = nBands
        self.fMin = fMin
        self.fMax = fMax
        self.fRef = fRef
        self.bax = bax
        self.fax = fax
        self.g = g
        self.d = d
        self.gDual = gDual
        self.type = type

        if type == "sparse":                                        # In Sparse mode store data
            idxs = [None] * nBands                                  # based on indexes after
            gList = [None] * nBands                                 # truncation into lists
            gDualList = [None] * nBands
            shifts = [None] * nBands

            for k in range(nBands):
                ii = np.where(g[k,:] != 0)[0]                       # Find valid indexes
                ii = NsgfCQT.padIdxs(ii)
                nCoefs = len(ii)
                offset = ii[0]
                idxs[k] = ii
                n = np.arange(nCoefs)
                shifts[k] = np.exp(1j * 2 * np.pi * offset * n / nCoefs)
                gList[k] = g[k,:]
                gDualList[k] = gDual[k,:]

            self.g = gList
            self.gDual = gDualList
            self.idxs = idxs
            self.shifts = shifts

    def forward(self, x):        
        X = fft.fft(x, self.nSamps) / self.nSamps

        if self.type == "full":
            Xcq = 2 * fft.ifft(X * self.g) * self.nSamps
        elif self.type == "sparse":
            Xcq = self.nBands * [None]
            for k in range(self.nBands):
                nCoefs = float(len(self.idxs[k]))
                Xi = fft.ifft(X[self.idxs[k]] * self.g[k])
                Xi = Xi * 2 * nCoefs * self.shifts[k]
                Xcq[k] = Xi
        return Xcq

    def inverse(self, Xcq):
        if self.type == "full":
            X = np.sum(fft.fft(Xcq) * self.gDual, 0) / (2.0 * self.nSamps)
        elif self.type == "sparse":
            X = np.zeros(self.nSamps, dtype=np.complex128)
            for k in range(self.nBands):
                nCoefs = len(self.idxs[k])
                Xi = Xcq[k]
                Xi *= self.shifts[k].conj() / (2.0 * nCoefs)
                Xi = fft.fft(Xi) * self.gDual[k]
                X[self.idxs[k]] += Xi
        x = fft.irfft(X, self.nSamps) * self.nSamps
        return x
    
    @staticmethod
    def padIdxs(ii):
        i0 = ii[0]
        nIdx = ii.size
        if nIdx < 4:
            nIdx = 4
        nidx = int(2**np.ceil(np.log2(nIdx)))
        ii = np.arange(i0, i0+nIdx-1)
        return ii
    
    def rasterize(self, Xcq):
        Xr = np.zeros([self.nBands, self.nSamps], dtype=np.complex128)
        for k in range(self.nBands):
            Xr[k,:] = sg.resample(Xcq[k], self.nSamps)
        return Xr
    
# function ii = padIdxs(ii)
#     i0 = ii(1);
#     nIdx = length(ii);
#     if nIdx < 4; nIdx = 4; end
#     nIdx = 2^nextpow2(nIdx);
#     ii = (i0:i0+nIdx-1);
# end