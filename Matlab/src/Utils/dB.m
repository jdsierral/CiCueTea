% DB  Magnitude in decibels
%   y = dB(z)
%
%   Converts (complex) amplitudes to decibels: y = 20*log10(abs(z)).
%   Included in the repo because Demo2 and domainColor depend on it —
%   previously it only existed on the author's personal MATLAB path.
%
%   See also: domainColor, mag2db

function y = dB(z)
    y = 20 * log10(abs(z));
end
