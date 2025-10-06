
% DOMAINCOLOR  Map a complex matrix to an RGB image using magnitude and phase
%   c = domainColor(z, dBLimits, angSteps, magSteps)
%
%   Maps the complex values in z to an RGB image using HSV color space:
%     - Hue encodes phase (angle)
%     - Value encodes magnitude (in dB, clipped and normalized)
%     - Saturation is always 1
%
%   INPUTS:
%     z        : Complex-valued input matrix
%     dBLimits : [min, max] dB range for magnitude normalization (default: [-60, 0])
%     angSteps : Number of discrete hue steps for phase (default: 360)
%     magSteps : Number of discrete value steps for magnitude (default: 1024)
%
%   OUTPUT:
%     c : RGB image (same height/width as z, 3 color channels)
%
%   See also: hsv2rgb, angle, dB

function c = domainColor(z, dBLimits, angSteps, magSteps)
    if (nargin < 4)
        magSteps = 2^10; % Number of quantization steps for magnitude
    end
    if (nargin < 3)
        angSteps = 360;  % Number of quantization steps for angle
    end
    if (nargin < 2)
        dBLimits = [-60, 0]; % dB range for normalization
    end

    % Convert magnitude to dB and get phase angle
    dBMag = dB(z);
    dBAng = angle(z);

    % Clip and normalize magnitude to [0, 1]
    dBMag = clip(dBMag, dBLimits(1), dBLimits(2));
    dBMag = (dBMag - dBLimits(1)) ./ diff(dBLimits);
    dBMag = floor(magSteps * dBMag) / magSteps; % Quantize

    % Map phase to hue in [0, 1]
    hue = (dBAng + pi) / (2 * pi);
    hue = floor(angSteps * hue) / angSteps; % Quantize
    sat = ones(size(hue));                 % Full saturation
    val = dBMag;                           % Value = normalized magnitude

    % Combine HSV channels and convert to RGB
    c = cat(3, hue, sat, val);
    c = hsv2rgb(c);
end