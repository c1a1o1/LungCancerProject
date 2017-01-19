curSampleFolder = uigetdir;
[ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
    getDCMFolderData( curSampleFolder );

%%
imtool3D(dcmArrayHU)
%%

outRegion = find(dcmArrayHU<=-1200);
region1 = find(dcmArrayHU>-1200 & dcmArrayHU<-700);
region2 = find(dcmArrayHU>=-700);

volBlock = zeros(size(dcmArrayHU));
volBlock(outRegion)=1;
volBlock(region1)=2;
volBlock(region2)=3;
%%

%used to obtain bounding box for bones
%   since lungs are surrounded by ribs, this is good choice
tissueRegion = find(dcmArrayHU>=500); 
[rV,cV,zV] = ind2sub(size(dcmArrayHU),tissueRegion);
minR = min(rV); maxR = max(rV);
minC = min(cV); maxC = max(cV);
minZ = min(zV); maxZ = max(zV);

dcmArraySeg = dcmArrayHU(minR:maxR,minC:maxC,minZ:maxZ);

imtool3D(dcmArraySeg);

%%

graythresh(dcmArrayHU)