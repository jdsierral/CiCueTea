clear
clc

addpath(genpath("../src"));

load handel.mat
fs = 48000;
x = resample(y, fs, Fs);

nSamps = length(x);
frac = 1;
fMap = @freq2erb;
fMin = 100;
fMax = 10000;
fRef = 1000;

sERB = nsgfVQTInit("full", fs, nSamps, fMap, 1/2, fMin, fMax, fRef);
sCQT = nsgfVQTInit("full", fs, nSamps, @log2, 1/12, fMin, fMax, fRef);


XERB = nsgfCQT(x, sERB);
XCQT = nsgfCQT(x, sCQT);

figure(1);
clf();
subplot 211
semilogx(sERB.fax, sERB.g, "LineWidth", 2); 
xlim([2e1, 2e4])
xlabel("Frequency");
ylabel("Amplitude");
title("ERB Based NSGF-VQT")

subplot 212
semilogx(sCQT.fax, sCQT.g, "LineWidth", 2);
xlim([2e1, 2e4])
xlabel("Frequency");
ylabel("Amplitude");
title("NSGF-CQT")

tax = sCQT.tax;
tax = tax - tax(1);

figure(2)
clf()
range = [-50, 0];
subplot 211
imagesc(tax, sERB.bax, dB(XERB).');
colormap jet
colorbar
set(gca, "CLim", range)
title("ERB Transform");
set(gca, "YDir", "normal");
xlabel("Time");
ylabel("ERBs");

subplot 212
imagesc(tax, sCQT.bax, dB(XCQT).');
colormap jet
colorbar
set(gca, "CLim", range)
title("CQT Transform");
set(gca, "YDir", "normal");
xlabel("Time");
ylabel("Log2 of Freq");