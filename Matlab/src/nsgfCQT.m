% NSGFCQT  Nonstationary Gabor Frame Constant-Q Transform
%   Xcq = nsgfCQT(x, s)
%
%   Computes the Constant-Q Transform (CQT) of the input signal x using a
%   nonstationary Gabor frame (NSGF) approach. The function supports both
%   dense and sparse representations depending on the structure s.
%
%   INPUTS:
%     x : Input signal (vector)
%     s : Structure containing transform parameters and precomputed values:
%         .nSamps   - Number of samples for FFT
%         .type     - "dense" or "sparse" (output type)
%         .g        - gabor frame (vector or cell array)
%         .nBands   - Number of frequency bands (for sparse type)
%         .idxs     - Cell array of frequency indices (for sparse type)
%         .phase    - Cell array of phase factors (for sparse type)
%
%   OUTPUT:
%     Xcq : Constant-Q transform coefficients. If s.type is "dense", Xcq is a
%           vector. If s.type is "sparse", Xcq is a cell array with one entry
%           per frequency band.
%
%   See also: nsgfCQTInit, nsgfICQT, nsgfRasterize

function Xcq = nsgfCQT(x, s)
    % Compute FFT of input and normalize
    X = fft(x, s.nSamps) ./ s.nSamps;
    
    if s.type == "dense"
        % Dense mode: apply all filters at once, then IFFT
        Xcq = 2 * s.nSamps * ifft(X .* s.g);
    elseif s.type == "sparse"
        % Sparse mode: process each band separately
        Xcq = cell(s.nBands, 1);
        for k = 1:s.nBands
            nCoefs = length(s.idxs{k});   % Number of coefficients for this band
            % Apply band filter in frequency domain, then IFFT
            Xi = ifft(X(s.idxs{k}) .* s.g{k});
            % Scale and apply precomputed phase correction
            Xi = 2 * Xi .* nCoefs .* s.phase{k};
            Xcq{k} = Xi;
        end
    end
end