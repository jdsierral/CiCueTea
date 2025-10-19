import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import scipy.fft as fft
import scipy.signal as sg
import control as ctl
import NsgfCQT
from Slicing import *
from psychoacoustics import *


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

sERB = NsgfCQT.NsgfVQT('dense', fs, nSamps, freq2erb, 1/2)
sCQT = NsgfCQT.NsgfVQT('dense', fs, nSamps, np.log2, 1/12)

Xerb = sERB.forward(x)
Xcqt = sCQT.forward(x)


plt.figure(1)
plt.clf()
plt.subplot(2, 1, 1)
plt.pcolormesh(sERB.time_axis, sERB.band_axis, dB(Xerb).T)
plt.clim(-100, 0)

plt.subplot(2, 1, 2)
plt.pcolormesh(sCQT.time_axis, sCQT.band_axis, dB(Xcqt).T)
plt.clim(-100, 0)



