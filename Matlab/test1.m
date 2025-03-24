clear
clc

fs = 48000;
N = 2^10;
ppo = 1;
fMin = 100;
fMax = 10000;
fRef = 1500;

s = nsgfCQTInit("full", fs, N, ppo, fMin, fMax, fRef);

t = (0:N-1)' / fs;
x = sin(2 * pi * fRef * t);

Xcq = nsgfCQT(x, s);
y = nsgfICQT(Xcq, s);

rms(y - x)

Xcq_ = csvread("/Users/juans/Developer/NYU/Thesis/CQTLib_Eigen/Build/Tests/Debug/Xcq.csv").';
% Xdft_ = csvread("/Users/juans/Developer/NYU/Thesis/CQTLib_Eigen/Build/Tests/Debug/Xdft.csv").';
% Xmat_ = csvread("/Users/juans/Developer/NYU/Thesis/CQTLib_Eigen/Build/Tests/Debug/Xmat.csv").';

rms(Xcq_ - Xcq, 'all')

subplot 211
imagesc(dB(Xcq));
colorbar

subplot 212
imagesc(dB(Xcq_));
colorbar
