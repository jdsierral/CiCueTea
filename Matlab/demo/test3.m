clear
clc

addpath(genpath("src"));

fs = 48000;
nSamps = 2^20;
frac = 1;
fMin = 100;
fMax = 10000;
fRef = 1000;

t = (0:nSamps-1)'/fs;
x = chirp(t, fMin, t(end), fMax, "logarithmic");
w = kaiser(nSamps, 20);
x = x .* w;

blockSize = 2^14;
overlapSize = blockSize / 2;
win = sqrt(hann(blockSize, "periodic"));

x(1:overlapSize) = 0;
x(end-overlapSize:end) = 0;

sLong  = nsgfCQTInit("full", fs, nSamps,    frac);
XcqRef = nsgfCQT(x, sLong);

s = nsgfCQTInit("full", fs, blockSize, frac);

% Slice the signal
xBuf = slicer(x, blockSize, overlapSize);
% Window the time domain blocks
xBuf = xBuf .* win;

% NSGF-CQT
XcqBuf = colfun(@(x) nsgfCQT(x, s), xBuf, 'uni', 0); XcqBuf = cat(3, XcqBuf{:});

% Window time-frequency planes
XcqBuf = XcqBuf .* win;

% ReBuild long Form
Xcq = spectralSplicer(XcqBuf, overlapSize);
% 
XcqBuf_ = spectralSlicer(Xcq, blockSize, overlapSize);

% Window time-frequency planes
XcqBuf_ = XcqBuf_ .* win;

% NSGF-ICQT
XcqBuf_ = squeeze(num2cell(XcqBuf_, [1, 2]));
xBuf_ = cellfun(@(x) nsgfICQT(x, s), XcqBuf_, 'uni', 0);
xBuf_ = [xBuf_{:}];

% Window the time blocks
xBuf_ = xBuf_ .* win;

% Combine the time blocks
x_ = splicer(xBuf_, overlapSize);


rms(XcqRef - Xcq, 'all')
rms (x - x_)

figure(1)
clf();
subplot 311
plot(t, x)
xlabel("Time");
ylabel("Amplitude");
title("Original Time-domain signal")

subplot 312
plot(t, x_)
xlabel("Time");
ylabel("Amplitude");
title("Reconstructed Time-domain signal")

subplot 313
plot(t, x - x_);
xlabel("Time");
ylabel("Amplitude");
title("Signal error")

figure(2)
clf();
range = [-60, 0];
subplot 211
imagesc(t, log2(sLong.bax), dB(Xcq.'));
set(gca, "CLim", range);
set(gca, "YDir", "normal");
colormap jet
colorbar
ticks = yticks();
yticklabels(2.^ticks);
xlabel("Time");
ylabel("Frequency");
title("CQT as Computed from spliced short NSGF-CQT")

subplot 212
imagesc(t, log2(sLong.bax), dB(XcqRef.'));
set(gca, "CLim", range);
set(gca, "YDir", "normal");
colormap jet
colorbar
ticks = yticks();
yticklabels(2.^ticks);
xlabel("Time");
ylabel("Frequency");
title("CQT as Computed from full signal")


