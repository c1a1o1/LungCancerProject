%chunkPred = load('submissionsFinal/KaggleNN_NN_Prediction_BestOne_MAT.mat');
%wholePred = load('submissionsFinal/VGG19PlusXGBoost_BestOne_MAT.mat');

chunkPred = load('submissionsFinal/Stage2Block64Prediction.mat');
wholePred = load('submissionsFinal/Stage2WholeScanPrediction.mat');

id = wholePred.id;

cancerOutput = (chunkPred.cancer).*0.7 + (wholePred.cancer).*0.3;

fileID = fopen('submissionsFinal/Stage2chunk70Whole30SplitSubmission.csv','w');
fprintf(fileID,'id,cancer\n');
for i = 1:length(id)
    fprintf(fileID,'%s,%d\n',id{i},cancerOutput(i));
end
fclose(fileID);