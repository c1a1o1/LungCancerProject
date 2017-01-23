load('stage1_validationIDs.mat');

load('stage1_labelsMAT.mat');
percentPositive = sum(labelData)/length(labelData);

numValid = length(id);
submit = zeros(numValid,1);
for i = 1:numValid
    submit(i) = (rand<percentPositive);
end

fileID = fopen('submission.csv','w');
fprintf(fileID,'id,cancer\n');
for i = 1:numValid
    fprintf(fileID,'%s,%d\n',id{i},submit(i));
end
fclose(fileID);