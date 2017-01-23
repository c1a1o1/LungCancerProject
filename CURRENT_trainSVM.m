

numTrainTestUse = 50;
numTrain = 40;


load('stage1_labelsMAT.mat');
load('stage1_validationIDs.mat');

validationIDs = id;
trainTestIDs = names;

numTrainTest = length(trainTestIDs);
numValid = length(validationIDs);

randomInds = randperm(numTrainTest);
indsUse = randomInds(1:numTrainTestUse);

Xdata = zeros(numTrainTestUse,256*256*100);
Ydata = zeros(numTrainTestUse,1);

Xvalid = zeros(numValid,256*256*100);

for jj = 1:numTrainTestUse
   currentID = trainTestIDs{indsUse(jj)}; %TEMPORARY COMMENTED. TO UNCOMMENT SOON
   %currentID = trainTestIDs{jj};
   curName = strcat('segFilesResizedAll/resizedSegDCM_',currentID,'.mat');
   load(curName);
   Xdata(jj,:)=(resizedDCM(:))';
   Ydata(jj,1)=labelData(jj,1);
end

for kk = 1:numValid
   currentID = validationIDs{jj};
   curName = strcat('segFilesResizedAll/resizedSegDCM_',currentID,'.mat');
   load(curName);
   Xvalid(jj,:)=(resizedDCM(:))';
end

trainTestScore = [];
validScore = [];
Inc = 66;
endIndex = Inc+1;
while(endIndex-Inc<numValid)
	if(endIndex < numValid)
		XvalidCur = Xvalid(endIndex-Inc:endIndex,:);
	else
		XvalidCur = Xvalid(endIndex-Inc:end,:);
	end
	
	XdataCur = [Xdata;XvalidCur];
	[~,curScore,~,~,~,~] = pca(XdataCur);
	trainTestScore = curScore(1:numTrainTestUse,:);
	curValidScore = curScore(numTrainTestUse+1:end,:);
	validScore = [validScore;curValidScore];
	endIndex = endIndex + Inc + 1;
	

end


Xtrain = Xdata(1:numTrain,:);
Ytrain = Ydata(1:numTrain,:);

Xtest = Xdata(numTrain+1:end,:);
Ytest = Ydata(numTrain+1:end,:);

XtrainScore = trainTestScore(1:numTrain,:);
XtestScore = trainTestScore(numTrain+1:end,:);

model = fitcsvm(XtrainScore,Ytrain);

YhatTrain = predict(model,XtrainScore);
YhatTest = predict(model,XtestScore);
YhatValid = predict(model,validScore);

trainError = sum(abs(YhatTrain-Ytrain))/length(Ytrain);
testError = sum(abs(YhatTest-Ytest))/length(Ytest);

save('svmResults1.mat','YhatValid','trainError','testError');







