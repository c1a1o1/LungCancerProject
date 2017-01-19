curSampleFolder = uigetdir;
[ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
    getDCMFolderData( curSampleFolder );

%%
imtool3D(dcmArrayHU)
%%

tissueRegion = find(dcmArrayHU>=500);
[rV,cV,zV] = ind2sub(size(dcmArrayHU),tissueRegion);
minR = min(rV); maxR = max(rV);
minC = min(cV); maxC = max(cV);
minZ = min(zV); maxZ = max(zV);

dcmArraySeg = dcmArrayHU(minR:maxR,minC:maxC,minZ:maxZ);
%%imtool3D(dcmArraySeg);

region1 = find(dcmArraySeg>-1200 & dcmArraySeg<-700);
binBlock = zeros(size(dcmArraySeg));
binBlock(region1)=1;

blockData = bwconncomp(binBlock);

%{
outRegion = find(dcmArrayHU<=-1200);
region1 = find(dcmArrayHU>-1200 & dcmArrayHU<-700);
region2 = find(dcmArrayHU>=-700);

volBlock = zeros(size(dcmArrayHU));
volBlock(outRegion)=1;
volBlock(region1)=2;
volBlock(region2)=3;

binBlock = zeros(size(dcmArrayHU));
binBlock(region1)=1;
%}


numBlocks = size(blockData.PixelIdxList,2);
sizes = zeros(1,numBlocks);
for i = 1:numBlocks
   sizes(i) = size(blockData.PixelIdxList{i},1); 
end
[numPixels,largestBlocks]=sort(sizes,'descend');

%used to obtain bounding box for bones
%   since lungs are surrounded by ribs, this is good choice
%tissueRegion = find(dcmArrayHU>=500); 
%tissueRegion = find(dcmArrayHU>=100);
tissueRegion = blockData.PixelIdxList{largestBlocks(1)};
[rV,cV,zV] = ind2sub(size(dcmArraySeg),tissueRegion);
minR = min(rV); maxR = max(rV);
minC = min(cV); maxC = max(cV);
minZ = min(zV); maxZ = max(zV);

dcmArraySeg2 = dcmArraySeg(minR:maxR,minC:maxC,minZ:maxZ);

imtool3D(dcmArraySeg2);

