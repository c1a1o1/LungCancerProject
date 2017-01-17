curSampleFolder = uigetdir;
[ dcmData,dcmArray,dcmArrayHU,slope,intercept,dcmInfo ] = ...
    getDCMFolderData( curSampleFolder );

%%
imtool3D(dcmArrayHU)
%%

w     = 2;       % bilateral filter half-width
sigma = [2 0.1]; % bilateral filter standard deviations

% Apply bilateral filter to each image.

dcmArrayFiltered = zeros(size(dcmArrayHU));

minFactor = min(dcmArrayHU(:));
divFactor = max(dcmArrayHU(:)) - minFactor;

dcmArrayHUScaled = (dcmArrayHU-minFactor)./(divFactor);
curSlice = zeros(size(dcmArrayHU,1),size(dcmArrayHU,2),1);
%for i = 1:size(dcmArrayHU,3)
for i=13:40
    fprintf('Slice %d\n',i);
    curSlice(:,:,1)=dcmArrayHUScaled(:,:,i);
    dcmArrayFiltered(:,:,i)=bfilter2(curSlice,w,sigma);
end

dcmArrayFilteredRedone = (dcmArrayFiltered.*divFactor) + minFactor;

%%
dcmArrayGoodPixels = double(dcmArrayHU>-1200).*double(dcmArrayHU<-800);
imtool3D(dcmArrayGoodPixels)
%%

graythresh(dcmArrayHU)