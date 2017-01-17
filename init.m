curSampleFolder = uigetdir;
[ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
    getDCMFolderData( curSampleFolder );

%%
imtool3D(dcmArrayHU)
%%

dcmArrayGoodPixels = double(dcmArrayHU>-1200).*double(dcmArrayHU<-800);
imtool3D(dcmArrayGoodPixels)
%%


%%
graythresh(dcmArrayHU)