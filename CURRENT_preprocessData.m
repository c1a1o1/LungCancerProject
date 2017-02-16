%dirName = 'data/sample_images';
dirName = 'data/stage1/stage1';

dcmFolders = dir(dirName);
numFolders = size(dcmFolders,1);


for i = 1:numFolders
    
    foldname = dcmFolders(i,1).name;
    curSampleFolder = strcat(dirName,'/',foldname);
    if(length(foldname) < 4)
       continue; %do not include it 
    end
    
    fprintf(strcat('Now processing file ',num2str(i),' of ',num2str(numFolders)));

    %curSampleFolder = uigetdir;
    [ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
        getDCMFolderData( curSampleFolder );

    [outputDCM,~]=segmentLungFromBlock(dcmArrayHU);

    foldName = curSampleFolder(end-31:end);
    
    %resizedDCM = imresize3d(outputDCM,[],[256 256 100],'nearest','fill');
    %newFileName = strcat('segFilesResizedAll/resizedSegDCM_',foldName);
    %save(newFileName,'resizedDCM');
    
    newFileName = strcat('segFiles/segDCM_',foldName);
    save(newFileName,'outputDCM');

end

