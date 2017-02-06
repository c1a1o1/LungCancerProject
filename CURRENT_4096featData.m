load('stage1_labelsMAT.mat');
numObs = 50;
inds = randperm(1390);
indsUse = inds(1:numObs)+2;
allData = zeros(numObs*100,4096);
allLabels = zeros(numObs*100,1);
files = dir('feats4096layer');

for kk = 1:numObs
    curFeatData = load(strcat('feats4096layer/feats2D_4096layer_mat_'...
        ,names{indsUse(kk)},'.mat'));
    
    startInd = ((kk-1)*100+1); endInd = kk*100;
    allData(startInd:endInd,:)=curFeatData.featureDataAlexNet;
    allLabels(startInd:endInd,1)=labelData(indsUse(kk));
end


%%
load('autoencodingPrediction.mat')

endT = 6;
for i = 1:endT
    startI = (i-1)*100 + 1;
    endI = i*100;
   curData = validData( startI:endI, :);
   figure
   imagesc(curData);
   colorbar;
end