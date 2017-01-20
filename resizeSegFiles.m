dirName = 'segFiles';

dcmFolders = dir(dirName);
numFolders = size(dcmFolders,1);

for i = 1:numFolders
    i
    filename = dcmFolders(i,1).name;
    currentFile = strcat(dirName,'/',filename);
    if(length(filename) < 4)
       continue; %do not include it 
    end
    
    endPartFileName = currentFile(17:end);
    
    load(currentFile);
    resizedDCM = imresize3d(outputDCM,[],[256 256 100],'nearest','fill');
    
    newFileName = strcat('segFilesResized/resizedSegDCM_',endPartFileName);
    save(newFileName,'resizedDCM');
end