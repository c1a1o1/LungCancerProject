curFeatData = zeros(64,64,100);

for i = 1:100
   curFeatData(:,:,i)=reshape(featureDataAlexNet(i,:),64,64); 
end

%%
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
[coeff,score,latent]=pca(allData);

scoreMat = score(:,1:10);
patientMatrix = zeros(numObs,1000);
patLabels = zeros(numObs,1);
for kk = 1:numObs
    startInd = ((kk-1)*100+1); endInd = kk*100;
    curPat = scoreMat(startInd:endInd,:);
    patientMatrix(kk,:)=curPat(:);
    patLabels(kk) = labelData(indsUse(kk));
end


[score2, coeff2, latent2] = pca(patientMatrix);

indsGood2 = find(patLabels==0);
indsBad2 = find(patLabels==1);
xData = score2(:,1);
yData = score2(:,2);
zData = score2(:,3);
figure
hold on
plot(xData(indsGood2),yData(indsGood2),'b.');
plot(xData(indsBad2),yData(indsBad2),'r.');
hold off