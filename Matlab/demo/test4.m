clear
clc

addpath("src");

fs = 48000;
nSamps = 2^18;
frac = 1;
fMin = 100;
fMax = 10000;
fRef = 1000;

t = (0:nSamps-1)'/fs;
x = chirp(t, fMin, t(end), fMax, "logarithmic");
w = kaiser(nSamps, 20);
x = x .* w;

s = nsgfCQTInit("sparse", fs, nSamps);
Xcq = nsgfCQT(x, s);

