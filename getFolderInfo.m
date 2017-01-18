dirName = 'data\\sample_images';
sampleImagesFolder = dir(dirName);

numFileSets = size(sampleImagesFolder,1);
dcmInfoArray = cell(1,numFileSets);
n = 4;
index = 1;
for i = 1:numFileSets
    filename = sampleImagesFolder(i,1).name;
    filepath = strcat(dirName,'\',filename);
    if(length(filename) < 4 )
       continue; %do not include it 
    end
    fprintf('Now processing patient %d of %d\n',index,numFileSets-2);
    [ ~,~,~,~,~,dcmInfoArray{index} ] = ...
        getDCMFolderData( filepath );
    
    index = index + 1;
end
dcmInfoArray = dcmInfoArray(1:(index-1));

