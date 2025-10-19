import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import scipy.fft as fft
import scipy.signal as sg
import control as ctl
import NsgfCQT
from Slicing import *
from psychoacoustics import *

def pow2db(x):
    return 10.0 * np.log10(np.abs(x))

def dB(X):
    return 20.0 * np.log10(np.abs(X) + 1e-10)

def rms(x):
    return np.sqrt(np.mean(np.abs(x)**2.0))

fs = 48000
nSamps = 2**18
frac = 1.0
fMin = 100
fMax = 10000

t = np.arange(nSamps) / fs
x = sg.chirp(t, fMin, t[-1], fMax, 'logarithmic')
w = sg.windows.kaiser(nSamps, 20)
x *= w

s1 = NsgfCQT.NsgfCQT('dense', fs, nSamps, frac)
s2 = NsgfCQT.NsgfCQT('sparse', fs, nSamps, frac)

X1 = s1.forward(x)
X2 = s2.forward(x)
X2 = s2.rasterize(X2)

plt.figure(1)
plt.clf()
plt.subplot(3, 1, 1)
plt.pcolormesh(s1.time_axis, s1.band_axis, dB(X1).T)
plt.yscale('log')
plt.clim(-100, 0)

plt.subplot(3, 1, 2)
plt.pcolormesh(s2.time_axis, s2.band_axis, dB(X2).T)
plt.yscale('log')
plt.clim(-100, 0)

plt.subplot(3, 1, 3)
plt.pcolormesh(s2.time_axis, s2.band_axis, dB(X1 - X2).T)
plt.yscale('log')
plt.clim(-100, 0)

g1 = s1.g[:s1.n_freqs//2+1,:]
h1 = fft.irfft(g1, s1.n_samples, 0)
h1 = fft.ifftshift(h1, 0)
h1 = h1 / np.max(np.abs(h1), 0)

h2 = np.zeros_like(h1)
for k in np.arange(s2.n_bands):
    g_i = np.zeros(s2.n_samples)
    g_i[s2.idxs[k]] = s2.g[k]
    g_i = g_i[:s2.n_freqs//2+1]
    h2[:,k] = fft.ifftshift(fft.irfft(g_i, s2.n_samples))
h2 = h2 / np.max(np.abs(h2), 0)

plt.figure(2)
plt.clf()
plt.subplot(2, 1, 1)
plt.semilogx(s1.band_axis, pow2db(np.mean(np.abs(X1)**2.0, 0)))
plt.semilogx(s2.band_axis, pow2db(np.mean(np.abs(X2)**2.0, 0)), "-.")
plt.legend(["dense", "sparse"])

plt.subplot(2, 1, 2)
plt.plot(s1.time_axis, pow2db(np.mean(np.abs(X1)**2.0, 1)))
plt.plot(s2.time_axis, pow2db(np.mean(np.abs(X2)**2.0, 1)), "-.")
plt.legend(["dense", "sparse"])

plt.figure(3)
plt.clf()
plt.subplot(1, 2, 1)
plt.plot(s1.time_axis, h1 + np.arange(s1.n_bands))
plt.xlim([-0.02, 0.02])

plt.subplot(1, 2, 2)
plt.plot(s2.time_axis, h2 + np.arange(s2.n_bands))
plt.xlim([-0.02, 0.02])