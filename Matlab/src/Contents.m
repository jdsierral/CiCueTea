% CICUETEA Real-time, invertible Constant-Q Transform (CQT) engine based on
% nonstationary Gabor frames (NSGF) — MATLAB reference implementation.
% Version 1.0.0 12-Jul-2026
%
% Constant-Q / Variable-Q transform
%   nsgfCQTInit     - Initialize a Nonstationary Gabor Frame for the CQT
%   nsgfVQTInit     - Initialize a Nonstationary Gabor Frame for a Variable-QT
%   nsgfCQT         - Forward transform (dense or sparse)
%   nsgfICQT        - Inverse transform (dense or sparse)
%   nsgfRasterize   - Rasterize sparse coefficients into the dense matrix form
%
% Slicing (block-streaming utilities)
%   slicer          - Slice a signal into overlapping blocks
%   splicer         - Reconstruct a signal from overlapping blocks (overlap-add)
%   spectralSlicer  - Slice a spectrogram into overlapping time blocks
%   spectralSplicer - Reconstruct a spectrogram from overlapping time blocks
%
% Utilities
%   dB              - Magnitude in decibels
%   domainColor     - Map a complex matrix to an RGB image (magnitude/phase)
%   ppoToQ          - Periods per octave to Q-factor
%   qToPpo          - Q-factor to periods per octave
%   colfun          - Apply a function to each column of a matrix
%
% See also: <a href="https://github.com/jdsierral/CiCueTea">github.com/jdsierral/CiCueTea</a>
