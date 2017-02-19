done=0;
dirName = 'rawHUdata';

dcmFolders = dir(dirName);
numFiles = size(dcmFolders,1);

numTimes = 0;
while(~done || numFiles<1597)
    fprintf(strcat('Loop has been run ',num2str(numTimes),' times\n'));
    numTimes = numTimes + 1;
    fprintf('Currently waiting 5 seconds to start again...\n');
    pause(5)
   try
       done=0; 
       CURRENT_preprocessDataFromMAT;
        done=1;
   catch
        done=0;
   end
end