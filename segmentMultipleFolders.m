dirName = 'data\\sample_images';

numDCMFolds = 0;
while(numDCMFolds < 20)
    
    dcmFolders = dir(dirName);
    numFolders = size(dcmFolders,1);
    
    numDCMFolds = 0;
    for i = 1:numFolders
    
        foldname = dcmFolders(i,1).name;
        if(length(foldname) < 4)
           continue; %do not include it 
        end
        numDCMFolds = numDCMFolds + 1;
    end
    
    fprintf(strcat('',num2str(numDCMFolds),...
        ' of 20 DCM folders uploaded. Waiting for all of them...\n'));
    pause(300);
end

for i = 1:numFolders
    
    foldname = dcmFolders(i,1).name;
    curSampleFolder = strcat(dirName,'\',foldname);
    if(length(foldname) < 4)
       continue; %do not include it 
    end
    

    %curSampleFolder = uigetdir;
    [ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
        getDCMFolderData( curSampleFolder );

    tissueRegion = find(dcmArrayHU>=500);
    [rV,cV,zV] = ind2sub(size(dcmArrayHU),tissueRegion);
    minR = min(rV); maxR = max(rV);
    minC = min(cV); maxC = max(cV);
    minZ = min(zV); maxZ = max(zV);

    dcmArraySeg = dcmArrayHU(minR:maxR,minC:maxC,minZ:maxZ);

    region1 = find(dcmArraySeg>-1200 & dcmArraySeg<-700);
    binBlock = zeros(size(dcmArraySeg));
    binBlock(region1)=1;

    blockData = bwconncomp(binBlock);

    numBlocks = size(blockData.PixelIdxList,2);
    sizes = zeros(1,numBlocks);
    for j = 1:numBlocks
       sizes(j) = size(blockData.PixelIdxList{j},1); 
    end
    [numPixels,largestBlocks]=sort(sizes,'descend');


    tissueRegion = blockData.PixelIdxList{largestBlocks(1)};
    [rV,cV,zV] = ind2sub(size(dcmArraySeg),tissueRegion);
    minR = min(rV); maxR = max(rV);
    minC = min(cV); maxC = max(cV);
    minZ = min(zV); maxZ = max(zV);

    dcmArraySeg2 = dcmArraySeg(minR:maxR,minC:maxC,minZ:maxZ);

    foldName = curSampleFolder(end-31:end);
    save(strcat('segmentedDCM_',foldName,'.mat'),'dcmArraySeg2');

end

