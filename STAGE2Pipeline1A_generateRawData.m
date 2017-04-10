%dirName = 'data/sample_images';
%dirName = 'data/stage1/stage1';
dirName = 'stage2Data/stage2/stage2';

dcmFolders = dir(dirName);
numFolders = size(dcmFolders,1);


for i = 159:numFolders
    
    clearvars -except dcmFolders numFolders dirName i
    
    foldname = dcmFolders(i,1).name;
    curSampleFolder = strcat(dirName,'/',foldname);
    if(length(foldname) < 4)
       continue; %do not include it 
    end
    
    fprintf(strcat('Now processing file ',num2str(i),' of ',num2str(numFolders),'\n'));

    %curSampleFolder = uigetdir;
    [ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
        getDCMFolderData( curSampleFolder );

    %[outputDCM,~]=segmentLungFromBlock(dcmArrayHU);

    foldName = curSampleFolder(end-31:end);
    
    %resizedDCM = imresize3d(outputDCM,[],[256 256 100],'nearest','fill');
    
    newFileName = strcat('rawHUdataStage2/rawDCM_',foldName);
    save(newFileName,'dcmArrayHU');
    

end

