clear

addpath("../src/");

fs = 48000;
nSamps = 2^16;
frac = 1/24;
fMin = 100;
fMax = 10000;
fRef = 1000;

t = (0:nSamps-1).'/fs;
x = chirp(t, fMin, t(end), fMax, "logarithmic");
win = kaiser(nSamps, 25);
x = x .* win;

s = nsgfCQTInit("full", fs, nSamps, frac, fMin, fMax, fRef);
X = nsgfCQT(x, s);

figure(1)
imagesc(domainColor(X.'));
set(gca, "YDir", "normal");