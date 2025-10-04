clear
clc

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
figure(1);
clf();
semilogx(sERB.fax, sERB.g); 
xlim([2e1, 2e4])

t = (0:nSamps-1)'/fs;
% x = chirp(t, fMin, t(end), fMax, "logarithmic");
% w = kaiser(nSamps, 20);
% x = x .* w;

XERB = nsgfCQT(x, sERB);
XCQT = nsgfCQT(x, sCQT);


figure(1)
clf()
range = [-60, 0];
subplot 211
imagesc(sERB.tax, sERB.bax, dB(XERB).');
colormap jet
colorbar
set(gca, "CLim", range)
title("ERB Transform");
set(gca, "YDir", "normal");

subplot 212
imagesc(sCQT.tax, sCQT.bax, dB(XCQT).');
colormap jet
colorbar
set(gca, "CLim", range)
title("CQT Transform");
set(gca, "YDir", "normal");