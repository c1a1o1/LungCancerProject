function [ minR,maxR,minC,maxC,minZ,maxZ ] = getBoundingBox( binaryArray )
%GETBOUNDINGBOX Summary of this function goes here
%   Detailed explanation goes here
inds = find(binaryArray);
[rV,cV,zV] = ind2sub(size(binaryArray),inds);
minR = min(rV); maxR = max(rV);
minC = min(cV); maxC = max(cV);
minZ = min(zV); maxZ = max(zV);

end

