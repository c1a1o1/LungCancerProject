function [ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfoArray ] = getDCMFolderData( dirName )
%GETDCMFOLDERDATA Summary of this function goes here
%   dcmData - puts the matrices into a cell
%   dcmAray - makes a large array

filesInFolder = dir(dirName);
numFiles = size(filesInFolder,1);
dcmData = cell(1,numFiles);
dcmInfoArray = cell(1,numFiles);
sliceLocations = zeros(1,numFiles);
suffix = '.dcm';
n = 4;
index = 1;
slope = 1;
intercept = 0;
for i = 1:numFiles
    filename = filesInFolder(i,1).name;
    filepath = strcat(dirName,'\',filename);
    if(length(filename) < 4)
       continue; %do not include it 
    end
    if(strcmp(filename(end-n+1:end), suffix))
       dcmData{index} = dicomread(filepath); 
       dcmInfo = dicominfo(filepath);
       dcmInfoArray{index} = dcmInfo;
       if(isfield(dcmInfo,'RescaleIntercept'))
           slope=dcmInfo.RescaleSlope;
           intercept=dcmInfo.RescaleIntercept;
       end
       if(isfield(dcmInfo,'SliceLocation'))
           sliceLocations(index) = dcmInfo.SliceLocation; 
       end
       index = index + 1;
    end
end
dcmData = dcmData(1:(index-1));
dcmInfoArray = dcmInfoArray(1:(index-1));
sliceLocations = sliceLocations(1:(index-1));

[~,sortedLocs] = sort(sliceLocations);
dcmData = dcmData(sortedLocs);
dcmInfoArray = dcmInfoArray(sortedLocs);

dcmArray = zeros([size(dcmData{1}) length(dcmData)]);
for k = 1:length(dcmData)
   dcmArray(:,:,k) = dcmData{k};
end

%does the Hounsfeld Unit conversion
dcmArrayHU = (dcmArray.*slope) + intercept;

end

