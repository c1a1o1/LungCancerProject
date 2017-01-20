function [ binBlock ] = getLargestComponentImage( dataBlock )
%GETLARGESTCOMPONENT Summary of this function goes here
%   Detailed explanation goes here

blockData = bwconncomp(dataBlock);
numBlocks = size(blockData.PixelIdxList,2);
sizes = zeros(1,numBlocks);
for i = 1:numBlocks
   sizes(i) = size(blockData.PixelIdxList{i},1); 
end
[~,largestBlocks]=sort(sizes,'descend');
compInds = blockData.PixelIdxList{largestBlocks(1)};
binBlock = zeros(size(dataBlock));
binBlock(compInds)=1;
end

