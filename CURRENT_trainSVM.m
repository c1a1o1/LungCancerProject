
numUsedTotal = 100;
numTrain = 80;


load('stage1_labelsMAT.mat');
numID = size(names,1);


randomInds = randperm(numID);
indsUse = randomInds(1:numUsedTotal);

Xdata = zeros(numUsedTotal,256*256*100);
Ydata = zeros(numUsedTotal,1);

for jj = 1:numUsedTotal
   %currentID = names{indsUse(jj)}; TEMPORARY COMMENTED. TO UNCOMMENT SOON
   currentID = names{jj};
   curName = strcat('segFilesResizedAll/resizedSegDCM_',currentID,'.mat');
   load(curName);
   Xdata(jj,:)=(resizedDCM(:))';
   Ydata(jj,1)=labelData(jj,1);
end

[coeff,score,latent] = pca(Xdata);

Xtrain = Xdata(1:numTrain,:);
Ytrain = Ydata(1:numTrain,:);

Xtest = Xdata(numTrain+1:end,:);
Ytest = Ydata(numTrain+1:end,:);

XtrainScore = score(1:numTrain,:);
XtestScore = score(numTrain+1,:);

model = fitcsvm(XtrainScore,Ytrain);

YhatTrain = predict(model,XtrainScore);
YhatTest = predict(model,XtestScore);

trainError = sum(abs(YhatTrain-Ytrain))/length(Ytrain)
testError = sum(abs(YhatTest-Ytest))/length(Ytest)







