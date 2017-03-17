function [ xSpacing,ySpacing,zSpacing ] = getDCMFolderInfo( dirName )
%GETDCMFOLDERDATA Summary of this function goes here
%   dcmData - puts the matrices into a cell
%   dcmAray - makes a large array

filesInFolder = dir(dirName);
numFiles = size(filesInFolder,1);
sliceLocations = zeros(1,numFiles);
suffix = '.dcm';
n = 4;
index = 1;
for i = 1:numFiles
    filename = filesInFolder(i,1).name;
    filepath = strcat(dirName,'/',filename);
    if(length(filename) < 4)
       continue; %do not include it 
    end
    if(strcmp(filename(end-n+1:end), suffix))
       dcmInfo = dicominfo(filepath);
       if(isfield(dcmInfo,'PixelSpacing'))
           xySpacing=dcmInfo.PixelSpacing;
           xSpacing = xySpacing(1);
           ySpacing = xySpacing(2);
       end
       if(isfield(dcmInfo,'SliceLocation'))
           sliceLocations(index) = dcmInfo.SliceLocation; 
       end
       index = index + 1;
    end
end
sliceLocations = sliceLocations(1:(index-1));

locs = sort(sliceLocations);
zSpacing = abs(locs(2)-locs(1));


end

