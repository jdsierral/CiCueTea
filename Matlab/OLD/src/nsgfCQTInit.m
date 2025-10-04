% NSGFCQTINIT creates a struct that holds the description of a NSGF_CQT
% transform
%
% Usage:
%   s = nsgfCQTInit(type, fs, nSamps, ppo, fMin, fMax, fRef)
% where:
%   type = "full" or "sparse"
%   fs = Sample Rate
%   nSamps = Number of Samples to process
%   ppo = Parts per Octave e.g. 3 for 1/3oct resolution (Default: 12)
%   fMin = Min Frequency of the transform (not necessarily exact) (Default: 1e2)
%   fMax = Max Frequency of the transform (not necessarily exact) (Default: 1e4)
%   fRef = Exact frequency at one of the bands. From here nBandsUp until above
%           of fMax and nBandsDn until below of fMin. (Default: 440)
%   th   = threshold for sparse version. A higher threshold is makes the sparse 
%           version more different than the full version but also more efficient
%           This doesn't change its invertibility.
%
% By JuanS - Apr 2025
%
% See also nsgfCQT, nsgfICQT, nsgfRasterize

function s = nsgfCQTInit(type, fs, nSamps, frac, fMin, fMax, fRef, th)
    if nargin < 8; th = 1e-6; end
    if nargin < 7; fRef = 1e3; end
    if nargin < 6; fMax = 1e4; end
    if nargin < 5; fMin = 1e2; end
    if nargin < 4; frac = 1/12; end

    if fMin == 0; error("Lower bound can't be 0"); end
    if fMax * 2 >= fs; error("Not respecting nyquist limit"); end
    if fMin * 2 > fMax; error("Leave at least 1 octave"); end
    if (fs / (fMin * (exp2(frac) - 1))) > nSamps; error("Q is too big"); end

    nBandsUp = ceil(1/frac * log2(fMax / fRef));
    nBandsDn = ceil(1/frac * log2(fRef / fMin));
    nBands = nBandsDn + nBandsUp + 1;       
    nFreqs = nSamps;
    bax = fRef * 2.^(frac * (-nBandsDn:nBandsUp)'); % band axis
    tax = (-nSamps/2:nSamps/2-1)' / fs;             % time axis
    fax = (0:nFreqs-1)' * fs / nFreqs;              % freq axis

    c = log(4) / (frac.^2);                         % horiz scale factor
    g = exp( -c * (log2(bax).' - log2(fax)).^2);    % analytic Gaussians
        
    g(fax <= bax(1), 1) = 1;                        % make the lowest a LPF
    g(fax >= bax(end), end) = 1;                    % make the highest a HPF
    
    if type == "sparse"                             % In sparse mode truncate
        g(g <= th) = 0;                             % gaussians
    end

    d = sum(g.^2, 2);
    gDual = g ./ d;                                 % compute the dualFrame

    assert(rms(sum(g .* gDual, 2) - 1) < 1e-10);    % check for invertibility

    g(fax > fs/2,:) = 0;                            % zero after nyqust to avoid
    gDual(fax > fs/2,:) = 0;                         % conjugate calculation

    s = struct("fs", fs, "nSamps", nSamps, "nFreqs", nFreqs, "nBands", nBands,...
                "fMin", fMin, "fMax", fMax, "fRef", fRef, "bax", bax, ...
                "fax", fax, "tax", tax, "g", g, "d", d, "gDual", gDual, ...
                "type", type);

    if type == "sparse"                             % In sparse mode store data
        idxs = cell(nBands, 1);                     % based on indexes after
        gCell = cell(nBands, 1);                    % truncation into cells
        gDualCell = cell(nBands, 1);
        phase = cell(nBands, 1);
        
        for k = 1:nBands
            ii = find(g(:,k) ~= 0);                 % find valid indexes
            ii = padIdxs(ii);                       % pad indexes to power of 2
            nCoefs = length(ii);
            offset = ii(1) - 1;
            idxs{k} = ii;
            n = (0:nCoefs-1).';
            phase{k} = exp(1j*2*pi*offset*n/nCoefs); % phase offsets
            gCell{k} = g(ii,k);
            gDualCell{k} = gDual(ii,k);
        end
        
        s.g = gCell;                                  % Discard full description
        s.gDual = gDualCell;                          % store only the indexed
        s.idxs = idxs;                                % values
        s.phase = phase;
    end
end

function ii = padIdxs(ii)
    i0 = ii(1);
    nIdx = length(ii);
    if nIdx < 4; nIdx = 4; end
    nIdx = 2^nextpow2(nIdx);
    ii = (i0:i0+nIdx-1).';
end