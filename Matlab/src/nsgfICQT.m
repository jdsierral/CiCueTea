% NSGFICQT  Inverse Nonstationary Gabor Frame Constant-Q Transform
%   x = nsgfICQT(Xcq, s)
%
%   Reconstructs the time-domain signal from its Constant-Q Transform (CQT)
%   coefficients using the Nonstationary Gabor Frame (NSGF) approach.
%   Supports both dense and sparse representations.
%
%   INPUTS:
%     Xcq : CQT coefficients (vector or cell array, depending on s.type)
%     s   : Structure with transform parameters and dual windows (see nsgfCQTInit)
%
%   OUTPUT:
%     x   : Reconstructed time-domain signal (vector)
%
%   See also: nsgfCQT, nsgfCQTInit

function x = nsgfICQT(Xcq, s)
    if s.type == "dense"
        % Dense mode: sum over all bands, apply dual window, then IFFT
        X = 1./ (2 * s.nSamps) * sum(fft(Xcq) .* s.gDual, 2);
    elseif s.type == "sparse"
        % Sparse mode: reconstruct each band and sum into frequency bins
        X = zeros(s.nFreqs, 1);
        for k = 1:s.nBands
            nCoefs = length(s.idxs{k});           % Number of coefficients for this band
            Xi = Xcq{k};                          % Band coefficients
            % Undo phase and scale
            Xi = Xi .* conj(s.phase{k}) / (2 * nCoefs);
            % Transform back to frequency domain and apply dual window
            Xi = fft(Xi) .* s.gDual{k};
            % Add contribution to the appropriate frequency bins
            X(s.idxs{k}) = X(s.idxs{k}) + Xi;
        end
    end
    % Inverse FFT to recover time-domain signal, scale appropriately
    x = ifft(X, s.nSamps, "symmetric") .* s.nSamps;
end