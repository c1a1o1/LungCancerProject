load('sampleImageLabels');


XmatrixFromSample = zeros(19,256*256*100);
YcolumnFromSample = zeros(19,1);
for jj = 1:size(sampleDataWithLabels,1)
    jj
   currentID = sampleDataWithLabels{jj,1};
   curName = strcat('segFilesResized/resizedSegDCM_',currentID,'.mat');
   load(curName);
   XmatrixFromSample(jj,:)=(resizedDCM(:))';
   
   YcolumnFromSample(jj,1)=sampleDataWithLabels{jj,2};
end
