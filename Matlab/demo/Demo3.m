%% DEMO3  Block-wise NSGF-CQT Analysis, Slicing, and Reconstruction
%   This demo shows how to perform block-wise NSGF-CQT analysis and synthesis
%   on a long signal, using slicing, windowing, and splicing. It compares the
%   result to a reference dense-length transform and visualizes the error.
%   Requires the CiCueTea library and MATLAB's Signal Processing Toolbox.

clear
clc

% Add source code to path
addpath(genpath("src"));

% Analysis parameters
fs = 48000;           % Sample rate
nSamps = 2^20;        % Number of samples
frac = 1;             % Frequency resolution (periods per octave)
fMin = 100;           % Minimum frequency
fMax = 10000;         % Maximum frequency      

% Generate a logarithmic chirp signal and apply a Kaiser window
t = (0:nSamps-1)'/fs;
x = chirp(t, fMin, t(end), fMax, "logarithmic");
w = kaiser(nSamps, 20);
x = x .* w;

blockSize = 2^14;
overlapSize = blockSize / 2;
win = sqrt(hann(blockSize, "periodic"));

% Zero initial and final because they require a previous or next block to 
% reconstruct properly
x(1:overlapSize) = 0;
x(end-overlapSize:end) = 0;

% Initialize dense long version NSGF-CQT and compute its transform
sLong  = nsgfCQTInit("dense", fs, nSamps,    frac);
XcqRef = nsgfCQT(x, sLong);

% Initialize short NSGF-CQT
s = nsgfCQTInit("dense", fs, blockSize, frac);

% Slice the signal    
xBuf = slicer(x, blockSize, overlapSize);
% Window the time domain blocks
xBuf = xBuf .* win;

% NSGF-CQT (it is wrapped around a cell conversion to make avoid a for-loop)
XcqBuf = colfun(@(x) nsgfCQT(x, s), xBuf, 'uni', 0); XcqBuf = cat(3, XcqBuf{:});

% Window time-frequency planes
XcqBuf = XcqBuf .* win;

% ReBuild long Form
Xcq = spectralSplicer(XcqBuf, overlapSize);

% Slice long form into short form 
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
title("CQT as Computed from dense signal")


