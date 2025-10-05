function c = domainColor(z, dBLimits, angSteps, magSteps)
if (nargin < 4)
    magSteps = 2^10;
end
if (nargin < 3)
    angSteps = 360;
end
if (nargin < 2)
    dBLimits = [-60, 0];
end

dBMag = dB(z);
dBAng = angle(z);

dBMag = clip(dBMag, dBLimits(1), dBLimits(2));
dBMag = (dBMag - dBLimits(1)) ./ diff(dBLimits);
dBMag = floor(magSteps * dBMag) / magSteps;

hue = (dBAng + pi) / (2 * pi);
hue = floor(angSteps * hue) / angSteps;
sat = ones(size(hue));
val = dBMag;

c = cat(3, hue, sat, val);
c = hsv2rgb(c);
end