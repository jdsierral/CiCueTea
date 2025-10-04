clear

addpath("../src/");

fs = 48000;
nSamps = 2^16;
frac = 1/3;
fMin = 100;
fMax = 10000;
fRef = 1000;

t = (0:nSamps-1).'/fs;
x = chirp(t, fMin, t(end), fMax, "logarithmic");
win = kaiser(nSamps, 25);
x = x .* win;

s1 = nsgfCQTInit("sparse", fs, nSamps, frac, fMin, fMax, fRef, 1e-6);
s2 = nsgfCQTInit("full", fs, nSamps, frac, fMin, fMax, fRef);
X1 = nsgfCQT(x, s1);
X1 = nsgfRasterize(X1, s1);
X2 = nsgfCQT(x, s2);

figure(1)
subplot 131
imagesc(domainColor(X1.'));
set(gca, "YDir", "normal");
colorbar
colormap jet
set(gca, "CLim", [-40, 0])

subplot 132
imagesc(domainColor(X2.'));
set(gca, "YDir", "normal");
colorbar
colormap jet
set(gca, "CLim", [-40, 0])

subplot 133
imagesc(dB(X1 - X2)');
set(gca, "YDir", "normal");
colorbar
colormap jet
set(gca, "CLim", [-40, 0])