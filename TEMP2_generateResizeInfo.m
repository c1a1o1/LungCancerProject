%dirName = 'data/sample_images';
dirName = 'data/stage1/stage1';

dcmFolders = dir(dirName);
numFolders = size(dcmFolders,1);

for i = 1:1
    
    %foldname = dcmFolders(i,1).name;
    foldname='bbf7a3e138f9353414f2d51f0c363561';
    curSampleFolder = strcat(dirName,'/',foldname);
    if(length(foldname) < 4)
       continue; %do not include it 
    end
    
    fprintf(strcat('Now processing file ',num2str(i),' of ',num2str(numFolders),'\n'));

    [ xs,ys,zs ,SLICE,DCM] = getDCMFolderInfo2( curSampleFolder );
    resizeTuple = [xs,ys,zs];
    
    foldName = curSampleFolder(end-31:end);
    
    newFileName = strcat('volResizeInfo/resizeTuple_',foldName);
    save(newFileName,'resizeTuple');
    

end

