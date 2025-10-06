import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import scipy.fft as fft
import scipy.signal as sg
import control as ctl
import NsgfCQT
from Slicing import *


def dB(X):
    return 20.0 * np.log10(np.abs(X))

def rms(x):
    return np.sqrt(np.mean(np.abs(x)**2.0))

fs = 48000
nSamps = 2**20
frac = 1.0
fMin = 100
fMax = 10000

blockSize = 2**14
overlapSize = blockSize//2
win = np.sqrt(sg.windows.hann(blockSize, False))

t = np.arange(nSamps) / fs
x = sg.chirp(t, fMin, t[-1], fMax, 'logarithmic')
w = sg.windows.kaiser(nSamps, 20)
x *= w
x[:blockSize] = 0
x[-blockSize:] = 0

sRef = NsgfCQT.NsgfCQT('full', fs, nSamps, frac)
XcqRef = sRef.forward(x)

s = NsgfCQT.NsgfCQT('full', fs, blockSize, frac)

xBuf = slicer(x, blockSize, overlapSize)
xBuf *= win
XcqShort = [s.forward(xBuf[i,:]) for i in range(xBuf.shape[0])]
XcqShort = np.stack(XcqShort, axis=0)
XcqShort = XcqShort.transpose((0, 2, 1))
XcqShort *= win
Xcq = spectral_splicer(XcqShort, overlapSize)
XcqShort_ = spectral_slicer(Xcq, blockSize, overlapSize)
XcqShort_ *= win
xBuf_ = [s.inverse(XcqShort_[i,:,:].T) for i in range(XcqShort_.shape[0])]
xBuf_ = np.stack(xBuf_, axis=0)
xBuf_ *= win
x_ = splicer(xBuf_, overlapSize)

err1 = rms(XcqRef - Xcq)
err2 = rms(x - x_)

plt.figure(1)
plt.clf()
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Original Time-domain signal")

plt.subplot(3, 1, 2)
plt.plot(t, x_)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Reconstructed Time-domain signal")

plt.subplot(3, 1, 3)
plt.plot(t, x - x_)
plt.tight_layout()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Signal error")

plt.figure(2)
plt.clf()
cRange = [-60, 0]
plt.subplot(2, 1, 1)
plt.pcolormesh(t, s.band_axis, dB(Xcq.T))
plt.yscale('log')
plt.clim(cRange)
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("CQT as Computed from spliced short NSGF-CQT")

plt.subplot(2, 1, 2)
plt.pcolormesh(t, s.band_axis, dB(XcqRef.T))
plt.yscale('log')
plt.clim(cRange)
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("CQT as Computed from full signal")

plt.tight_layout()
