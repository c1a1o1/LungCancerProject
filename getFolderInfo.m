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

%%
load('sampleImageMetadata.mat')
%%
numRows = zeros(1,20);
numCol = zeros(1,20);
numSlice = zeros(1,20);
sizeZ = zeros(1,20);

pixelSpace1 = zeros(1,20);
pixelSpace2 = zeros(1,20);
for i = 1:20
    numRows(i) = dcmInfoArray{i}{1}.Rows;
    numCol(i) = dcmInfoArray{i}{1}.Columns;
    numSlice(i)=size(dcmInfoArray{i},2);
    sizeZ(i) = abs(dcmInfoArray{i}{1}.SliceLocation-...
        dcmInfoArray{i}{end}.SliceLocation);
    
    pixelSpacing = dcmInfoArray{i}{1}.PixelSpacing;
    pixelSpace1(i) = pixelSpacing(1);
    pixelSpace2(i) = pixelSpacing(2);
end