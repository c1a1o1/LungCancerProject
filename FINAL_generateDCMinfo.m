startDir = 'LIDC/PatientData/DOI/';
%%
%load('TEMP_dirsCheck.mat');
%%

for row = 1:size(dirsToCheck,1)
    row
    dirNameTail = dirsToCheck(row,32:end);
    folders = strsplit(dirNameTail,'/');
    folderName = folders{end};
   currentDir = strcat(startDir,dirNameTail);
   dcmFiles = dir(strcat(currentDir,'/*.dcm'));
   dcmArray = cell(1,length(dcmFiles));
    for ii = 1:length(dcmFiles)
        fullDir = strcat(currentDir,'/',dcmFiles(ii).name);
        dcmArray{ii} = dicominfo(fullDir);
    end
    
    saveFile = strcat('C:\dev\git\LungCancerProject\DOI_dcmInfo\dcmInfoArray',...
        folderName,'.mat');
    save(saveFile,'dcmArray')
end

