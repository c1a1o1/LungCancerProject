function [ outputDCM,finalLungRegion ] = segmentLungFromBlock( dcmArrayHU )
%SEGMENTLUNGFROMBLOCK Summary of this function goes here
%   Detailed explanation goes here


%{
Step 1: 
Get a bounding box around bone pixels (>500 HU units)
Since lungs will only be around the ribs
%}
fprintf('Obtaining bounding box around bone...\n');
[ minR,maxR,minC,maxC,minZ,maxZ ] = getBoundingBox( (dcmArrayHU>=500) );
insideBlock = zeros(size(dcmArrayHU));
insideBlock(minR:maxR,minC:maxC,minZ:maxZ)=1;
outsideBlock = find(insideBlock==0);

%{
Step 2: 
Split data into 3 groups
i) outside region: HU unit below -1500, represents no data/air
                    also part of outside block from Step 1
ii) lung region: possible lung region values
iii) other tissue region: represents tissue surrounding the lungs
%}

fprintf('Bounding box obtained. Separating into 3 regions...\n');
outRegion = find(dcmArrayHU<=-1200);
lungRegion = find(dcmArrayHU>-1200 & dcmArrayHU<-700);
otherTissueRegion = find(dcmArrayHU>=-700);

volBlock = zeros(size(dcmArrayHU));
volBlock(outRegion)=1;
volBlock(lungRegion)=2;
volBlock(otherTissueRegion)=3;
volBlock(outsideBlock)=1;

%{
Step 3: 
For each slice, find the largest connected component that is part
    of the "other tissue region"
Finding the largest connected component is a noise reduction tool
    that I am employing here
%}
fprintf('Separation complete\n');
fprintf('Now finding largest outside tissue component of each slice...\n');
%finds the largest connected component of tissue
binBlock1 = zeros(size(volBlock));
binBlock2 = zeros(size(volBlock));
binBlock1(otherTissueRegion)=1;
for slice = 1:size(binBlock1,3) %gets largest blob in each slice
    curSlice = binBlock1(:,:,slice);
    newSlice = getLargestComponentImage(curSlice);
    binBlock2(:,:,slice)=newSlice;
    fprintf(strcat('Finished processing slice ',num2str(slice),...
        ' of ',num2str(size(binBlock1,3)),'\n'));
end

%{
Step 4:
Select the "interior" of the other tissue.

For each row, we find the min and max col for the surrounding tissue
We then only select lung tissue pixels in that range
%}

%the lung will be part of region1. 
%It will be on the interior of region2
fprintf('Largest slice components found\n');
fprintf('Now finding interior of each slice\n');
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
   fprintf(strcat('Finished processing slice ',num2str(slice),...
        ' of ',num2str(size(binBlock3,3)),'\n'));
end

%{
Step 5:
Of the binary image created in Step 4 of possible lung pixels,
    select the largest component of 1's as the lung pixels
%}
fprintf('Interiors Found\n');
fprintf('Now Finding Largest Component in 3D block\n');
finalLungRegion = getLargestComponentImage(binBlock3);

fprintf('Component Found. Finished Segmenting Image\n');

[ minR,maxR,minC,maxC,minZ,maxZ ] = getBoundingBox( (finalLungRegion==1) );

outputDCM = dcmArrayHU(minR:maxR,minC:maxC,minZ:maxZ);
end

