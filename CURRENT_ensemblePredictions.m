chunkPred = load('submissionsFinal/KaggleNN_NN_Prediction_BestOne_MAT.mat');
wholePred = load('submissionsFinal/VGG19PlusXGBoost_BestOne_MAT.mat');

id = wholePred.id;

cancerOutput = (chunkPred.cancer).*0.2 + (wholePred.cancer).*0.8;

fileID = fopen('submissionsFinal/chunk20Whole80SplitSubmission.csv','w');
fprintf(fileID,'id,cancer\n');
for i = 1:length(id)
    fprintf(fileID,'%s,%d\n',id{i},cancerOutput(i));
end
fclose(fileID);