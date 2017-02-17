done=0;
dirName = 'rawHUdata';

dcmFolders = dir(dirName);
numFiles = size(dcmFolders,1);

while(~done || numFiles<1597)
   try
       done=0; 
       CURRENT_preprocessDataFromMAT;
        done=1;
   catch
        done=0;
   end
end