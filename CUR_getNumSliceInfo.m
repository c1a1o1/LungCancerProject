segs = dir('segFiles');

numR = zeros(1,length(segs));
numC = zeros(1,length(segs));
numSlic = zeros(1,length(segs));
for ii = 1:length(segs)
    ii
   fileNm = segs(ii).name;
   if(length(fileNm)<3)
       continue
   end
   load(strcat('segFiles/',fileNm));
   [ro,co,sli] = size(outputDCM);
   numR(ii)=ro;
   numC(ii)=co;
   numSlic(ii)=sli;
end

save('segFilesSlicInfo.mat','numR','numC','numSlic');