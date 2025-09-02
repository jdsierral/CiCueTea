function X = SpecDebuffering(XBuf, blockSize, overlapSize, nBlocks)

if iscell(XBuf)
    overlapRatio = overlapSize / blockSize;
    nBands = size(XBuf{1}, 1);
    X = cell(nBands, 1);
    % for k = 1:nBands
    %     Xi = cell2mat(XBuf(k,:)').';
    %     blockSize = size(Xi, 1);
    %     overlapSize = blockSize * overlapRatio;
    %     X{k} = deBuffering(Xi, blockSize, overlapSize).';
    % end
    for k = 1:nBands
        Xi = cellfun(@(x) x{k}.', XBuf, 'uni', 0);
        Xi = [Xi{:}];
        blockSize_k = size(Xi, 1);
        overlapSize_k = blockSize_k * overlapRatio;
        X{k} = deBuffering(Xi, blockSize_k, overlapSize_k).';
    end
else
    assert(size(XBuf, 2) == blockSize);
    [nBands, blockSize, nBlocks] = size(XBuf);
    hopSize = blockSize - overlapSize;
    nSamps = hopSize * nBlocks + overlapSize;
    
    X = zeros(nBands, nSamps);
    for k = 1:nBands
        X(k,:) = deBuffering(squeeze(XBuf(k,:,:)), blockSize, overlapSize);
    end
end

end