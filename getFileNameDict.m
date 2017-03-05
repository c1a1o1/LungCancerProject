function [ dcmFileInfoMap ] = getFileNameDict( dcmInfoFiles,startInd,numEndCut)
%GETFILENAMEDICT Summary of this function goes here
%   Detailed explanation goes here

dcmFileNames = cell(1,length(dcmInfoFiles));
dcmFilePatIDs = cell(1,length(dcmInfoFiles));
for ii = 1:length(dcmInfoFiles)
    currentFileName = dcmInfoFiles(ii).name;
    currentPatId = currentFileName(startInd:(end-numEndCut));
    dcmFileNames{ii} = currentFileName;
    dcmFilePatIDs{ii} = currentPatId;
end
dcmFileInfoMap = containers.Map(dcmFilePatIDs,dcmFileNames);

end

