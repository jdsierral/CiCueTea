clear
clc

fs = 48000;
N = 2^16;
ppo = 1;
fMin = 100;
fMax = 10000;
fRef = 1500;

blockSize = 2^12;
hopSize = blockSize / 2;
overlapSize = blockSize - hopSize;
win = hann(blockSize, "periodic");
win = sqrt(win);
% win = win * sqrt(2)/4;

sShort = nsgfCQTInit("full", fs, blockSize, ppo, fMin, fMax, fRef);
sLong = nsgfCQTInit("full", fs, N, ppo, fMin, fMax, fRef);

t = (0:N-1)' / fs;
x = sin(2 * pi * fRef * t);
fScale = 1.2;
x = chirp(t, fMin * fScale, t(end), fMax / fScale, "logarithmic");
x(1:blockSize) = 0;
x(end-blockSize+1:end) = 0;
% x = x .* kaiser(N, 20);
x = x .* hann(N, "periodic");

xBuf = buffering(x, blockSize, overlapSize);
Xcq = colfun(@(xi) nsgfCQT(xi, sShort), xBuf .* win, 'uni', 0);
Xcq = cellfun(@(x) x .* win.', Xcq, 'uni', 0);
X = cellDeBuffering(Xcq, blockSize, overlapSize);

x_ = csvread("/Users/juans/Developer/NYU/Thesis/CQTLib_Eigen/Build/Tests/Debug/Xvec.csv");
X_ = csvread("/Users/juans/Developer/NYU/Thesis/CQTLib_Eigen/Build/Tests/Debug/Xmat.csv").';

rms(x - x_)
rms(X - X_, 'all')

figure(1)
clf();
subplot 211
imagesc(domainColor(X))
set(gca, "YDir", "normal");

subplot 212
imagesc(domainColor(X_))
set(gca, "YDir", "normal");


figure(2); 
subplot 311; 
plot(x); 
subplot 312; 
plot(x_); 
subplot 313; 
plot(x - x_)