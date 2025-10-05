function result = colfun(func, A, varargin)
% COLFUN applies a function to each column of a matrix.
% 
% Usage:
%   result = colfun(@someFunction, A)
%   result = colfun(@someFunction, A, 'UniformOutput', true)
%
% Inputs:
%   - func: Function handle to apply to each column
%   - A: Input matrix
%   - varargin: Additional arguments for cellfun (e.g., 'UniformOutput', false)
%
% Output:
%   - result: Cell array or matrix depending on 'UniformOutput' setting

if ~ismatrix(A)
    error('Input A must be a 2D matrix.');
end

% Convert matrix columns into a cell array
C = mat2cell(A, size(A,1), ones(1, size(A,2)));

% Apply function to each column using cellfun
result = cellfun(func, C, varargin{:});
end
