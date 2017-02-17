%dirName = 'data/sample_images';
dirName = 'rawHUdata';

dcmFolders = dir(dirName);
numFiles = size(dcmFolders,1);

for i = 1:numFiles
    
    filename = dcmFolders(i,1).name;
    curFile = strcat(dirName,'/',filename);
    if(length(filename) < 4)
       continue; %do not include it 
    end
    
    fprintf(strcat('Now processing file ',num2str(i),' of ',num2str(numFiles),'\n'));
    
    newFileName = curFile(end-35:end);
    newFileName = strcat('segFiles/segDCM_',newFileName);
    
    if(exist(newFileName,'file')>0)
       fprintf('File was already processed. Moving onto next one\n'); 
       continue 
    end
    
    load(curFile);
    [outputDCM,~]=segmentLungFromBlock(dcmArrayHU);
    save(newFileName,'outputDCM');

end

