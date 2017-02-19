myFiles = dir('rawHUdata');


names = cell(1,length(myFiles));
for ii = 1:length(myFiles)
   names{ii} = myFiles(ii).name; 
end
allFileNames = containers.Map(names,ones(1,length(myFiles)));
allFileNames2 = containers.Map(names,ones(1,length(myFiles)));

%save('rawHUdataFiles.mat','allFileNames');

load('rawHUdataFiles.mat')

numF = 0;
for ii = 1:length(names)
   if(~isKey(allFileNames,names{ii}))
       numF = numF+1;
       copyfile(strcat('rawHUdata/',names{ii}),strcat('rawHUdataSend/',names{ii}));
   end
end