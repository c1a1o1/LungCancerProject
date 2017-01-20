curSampleFolder = uigetdir;
[ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
    getDCMFolderData( curSampleFolder );

%imtool3D(dcmArrayHU)

tissueRegion = find(dcmArrayHU>=500);
[rV,cV,zV] = ind2sub(size(dcmArrayHU),tissueRegion);
minR = min(rV); maxR = max(rV);
minC = min(cV); maxC = max(cV);
minZ = min(zV); maxZ = max(zV);

dcmArraySeg = dcmArrayHU(minR:maxR,minC:maxC,minZ:maxZ);
%%
imtool3D(dcmArraySeg);

%%

outRegion = find(dcmArraySeg<=-1200);
lungRegion = find(dcmArraySeg>-1200 & dcmArraySeg<-700);
otherTissueRegion = find(dcmArraySeg>=-700);

volBlock = zeros(size(dcmArraySeg));
volBlock(outRegion)=1;
volBlock(lungRegion)=2;
volBlock(otherTissueRegion)=3;

%finds the largest connected component of tissue
binBlock1 = zeros(size(volBlock));
binBlock2 = zeros(size(volBlock));
binBlock1(otherTissueRegion)=1;
for slice = 1:size(binBlock1,3) %gets largest blob in each slice
    curSlice = binBlock1(:,:,slice);
    blockData = bwconncomp(curSlice);
    numBlocks = size(blockData.PixelIdxList,2);
    sizes = zeros(1,numBlocks);
    for i = 1:numBlocks
       sizes(i) = size(blockData.PixelIdxList{i},1); 
    end
    [~,largestBlocks]=sort(sizes,'descend');
    tissueRegion1 = blockData.PixelIdxList{largestBlocks(1)};
    
    newSlice = zeros(size(curSlice));
    newSlice(tissueRegion1)=1;
    binBlock2(:,:,slice)=newSlice;
    fprintf(strcat('Finished processing slice ',num2str(slice),...
        ' of ',num2str(size(binBlock1,3)),'\n'));
end


%%
%the lung will be part of region1. 
%It will be on the interior of region2
binBlock3 = zeros(size(volBlock));
for slice = 1:size(binBlock3,3)
   for row = 1:size(binBlock3,1)
        curRow1 = volBlock(row,:,slice);
        curRow2 = binBlock2(row,:,slice);
        leftBound = find(curRow2==1, 1 );
        rightBound = find(curRow2==1, 1, 'last' );
        binBlock3(row,leftBound:rightBound,slice)=1;
        binBlock3(row,:,slice)=...
            binBlock3(row,:,slice).*(curRow1==2);
   end
end

imtool3D(binBlock3)
%%
%binBlock3 now gives us possible lung pixels
%   the largest connected component will be correct
blockData = bwconncomp(binBlock3);
numBlocks = size(blockData.PixelIdxList,2);
sizes = zeros(1,numBlocks);
for i = 1:numBlocks
   sizes(i) = size(blockData.PixelIdxList{i},1); 
end
[~,largestBlocks]=sort(sizes,'descend');
finalLungRegion = blockData.PixelIdxList{largestBlocks(1)};
%%
binBlock4 = zeros(size(binBlock3));
binBlock4(finalLungRegion)=1;


%%
%{
dcmArraySegFilt = zeros(size(dcmArraySeg));
for i = 1:size(dcmArraySegFilt,3)
   dcmArraySegFilt(:,:,i)=medfilt2(dcmArraySeg(:,:,i),[5 5],'zeros'); 
end
%}

lungRegion = find(dcmArraySeg>-1200 & dcmArraySeg<-700);
binBlock = zeros(size(dcmArraySeg));
binBlock(lungRegion)=1;

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

