%% DEMO1  Example: NSGF-VQT and NSGF-CQT Analysis and Visualization
%   This demo loads an audio signal, computes its ERB-based and CQT-based
%   nonstationary Gabor frame transforms, and visualizes the results.
%   Requires the CiCueTea library and MATLAB's Signal Processing Toolbox.

clear
clc


% Add source code to path
addpath(genpath("../src"));


% Load example audio (Handel)
load handel.mat
fs = 48000;                      % Target sample rate
x = resample(y, fs, Fs);         % Resample to target rate


% Analysis parameters
nSamps = length(x);              % Number of samples
fracCQT = 1;                     % Split in 1/12 of octaves
fracERB = 1/2;                   % split in fractions of half ERBs
fMap = @freq2erb;                % Frequency mapping for ERB transform
fMin = 100;                      % Minimum frequency
fMax = 10000;                    % Maximum frequency
fRef = 1000;                     % Reference frequency


% Initialize ERB-based and CQT-based transforms
sERB = nsgfVQTInit("dense", fs, nSamps, fMap, fracCQT, fMin, fMax, fRef);
sCQT = nsgfVQTInit("dense", fs, nSamps, @log2, fracERB, fMin, fMax, fRef);


% Compute NSGF-VQT and NSGF-CQT transforms
XERB = nsgfCQT(x, sERB);
XCQT = nsgfCQT(x, sCQT);


% Plot Transform Frames
figure(1);
clf();
subplot 211
semilogx(sERB.fax, sERB.g, "LineWidth", 2); % ERB Frame
xlim([2e1, 2e4])
xlabel("Frequency");
ylabel("Amplitude");
title("ERB Based NSGF-VQT")


subplot 212
semilogx(sCQT.fax, sCQT.g, "LineWidth", 2); % CQT Frame
xlim([2e1, 2e4])
xlabel("Frequency");
ylabel("Amplitude");
title("NSGF-CQT")


% Prepare time axis for spectrograms
tax = sCQT.tax;
tax = tax - tax(1);


% Plot time-frequency representations
figure(2)
clf()
range = [-50, 0]; % dB color range
subplot 211
imagesc(tax, sERB.bax, dB(XERB).'); % ERB spectrogram
colormap jet
colorbar
set(gca, "CLim", range)
title("ERB Transform");
set(gca, "YDir", "normal");
xlabel("Time");
ylabel("ERBs");


subplot 212
imagesc(tax, sCQT.bax, dB(XCQT).'); % CQT spectrogram
colormap jet
colorbar
set(gca, "CLim", range)
title("CQT Transform");
set(gca, "YDir", "normal");
xlabel("Time");
ylabel("Log2 of Freq");