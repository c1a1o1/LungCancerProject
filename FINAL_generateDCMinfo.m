%{
startDir = 'LIDC/PatientData/DOI/';
load('TEMP_dirsCheck.mat');
%}

dirCheck = 'D:\dev\git\LungCancerProject\randFiles';

allFolders = genpath(dirCheck);

folderListRaw = textscan(allFolders,'%s','Delimiter',';');
folderList = folderListRaw{1};

numFolders = length(folderList);
for fInd = 1:numFolders
    
    currentFolder = folderList{fInd};
   
    dcmFiles = dir(strcat(currentFolder,'\*.dcm'));
    if(~isempty(dcmFiles))
       
        dcmArray = cell(1,length(dcmFiles));
        for ii = 1:length(dcmFiles)
            fullDir = strcat(currentFolder,'\',dcmFiles(ii).name);
            dcmArray{ii} = dicominfo(fullDir);
        end
        
        saveFile = strcat('D:\dev\git\LungCancerProject\DOI_dcmInfo\dcmInfoArray_',...
            dcmArray{1}.SeriesInstanceUID,'.mat');
        save(saveFile,'dcmArray')

    end
    
end

%{
dirsToCheck = {'D:\dev\git\LungCancerProject\randFiles\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'};

for row = 1:size(dirsToCheck,1)
    row
    %{
    dirNameTail = dirsToCheck(row,32:end);
    folders = strsplit(dirNameTail,'/');
    folderName = folders{end};
   currentDir = strcat(startDir,dirNameTail);
    %}
   currentDir = dirsToCheck{row};
   dcmFiles = dir(strcat(currentDir,'/*.dcm'));
   dcmArray = cell(1,length(dcmFiles));
    for ii = 1:length(dcmFiles)
        fullDir = strcat(currentDir,'/',dcmFiles(ii).name);
        dcmArray{ii} = dicominfo(fullDir);
    end
    
    saveFile = strcat('D:\dev\git\LungCancerProject\DOI_dcmInfo\dcmInfoArray',...
        folderName,'.mat');
    save(saveFile,'dcmArray')
end
%}
