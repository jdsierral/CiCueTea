
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import scipy.signal as sg
import control as ctl
from NsgfCQT import NsgfCQT


def dB(X):
    return 20.0 * np.log10(np.abs(X))

fs = 48000
nSamps = 2**12
fMin = 100
fMax = 10000
fRef = 1500
frac = 1

x = np.random.randn(nSamps)

fScale = 1.1
f0 = fMin * fScale
f1 = fMax / fScale
t = np.arange(nSamps) / fs
x = sg.chirp(t, f0, t[-1], f1, method='logarithmic')
w = sg.windows.kaiser(nSamps, 25)
x = x * w
# x = np.sin(2 * np.pi * fRef * t)

print(x.shape)

s = NsgfCQT("sparse", fs, nSamps, frac, fMin, fMax, fRef)
X = s.forward(x)
y = s.inverse(X)

print(np.sqrt(np.mean((x-y)**2)))

if s.type == "sparse":
    X = s.rasterize(X)

Sxx = np.mean(np.abs(X)**2.0, 1)

plt.figure(1)
plt.clf()
plt.subplot(4, 1, 1)
plt.plot(x)
plt.subplot(4, 1, 2)
plt.imshow(dB(X), aspect='auto', origin='lower')
plt.colorbar()
plt.clim([-60, 0])
plt.subplot(4, 1, 3)
plt.plot(y)

plt.subplot(4, 1, 4)
plt.semilogx(s.bax, dB(Sxx))
plt.yticks(np.arange(-120, 6, 12))
plt.xlim([1e2, 1e4])
plt.ylim([-120, 12])
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.clf()
if s.type == "full":
    plt.subplot(3, 1, 1)
    plt.semilogx(s.fax, s.g.T)
    plt.subplot(3, 1, 2)
    plt.semilogx(s.fax, dB(s.d))
    plt.subplot(3, 1, 3)
    plt.semilogx(s.fax, s.gDual.T)
else:
    for k in range(s.nBands):
        plt.subplot(3, 1, 1)
        plt.semilogx(s.fax[s.idxs[k]], s.g[k])
        plt.subplot(3, 1, 2)
        plt.semilogx(s.fax, dB(s.d))
        plt.subplot(3, 1, 3)
        plt.semilogx(s.fax[s.idxs[k]], s.gDual[k])

plt.show()