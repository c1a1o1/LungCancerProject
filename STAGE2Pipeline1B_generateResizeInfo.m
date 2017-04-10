dirName = 'stage2Data/stage2/stage2';

dcmFolders = dir(dirName);
numFolders = size(dcmFolders,1);


for i = 1:numFolders
    
    foldname = dcmFolders(i,1).name;
    curSampleFolder = strcat(dirName,'/',foldname);
    if(length(foldname) < 4)
       continue; %do not include it 
    end
    
    fprintf(strcat('Now processing file ',num2str(i),' of ',num2str(numFolders),'\n'));

    [ xs,ys,zs ] = getDCMFolderInfo( curSampleFolder );
    resizeTuple = [xs,ys,zs];
    
    foldName = curSampleFolder(end-31:end);
    
    newFileName = strcat('volResizeInfoStage2/resizeTuple_',foldName);
    save(newFileName,'resizeTuple');
    

end

