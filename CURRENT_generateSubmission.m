
%load('RES_randomProj.mat');
load('cnnPredictionFrom_2017_01_27__17_13_13.mat');
%%
[XX,YY,TT] = perfcurve(Ytest,yHatTestP(:,2),1,...
    'XCrit','accu','YCrit','fpr');
[XX2,YY2,TT2] = perfcurve(Ytest,yHatTestP(:,2),1,...
    'XCrit','fnr','YCrit','fpr');

figure
hold on
plot(TT,XX,'k-')
plot(TT,YY,'r-')
plot(TT2,XX2,'b-')
hold off
legend('Accu vs Thresh','False Pos Vs Thresh','False Neg vs Thresh')

%%
%based on above, I will say 0.3 is good threshold
%thresh=0.3;
%submit = int16(YvalidP(:,2)>0.3);
submit = YvalidP(:,2);

load('stage1_validationIDs.mat');
load('stage1_labelsMAT.mat');

fileID = fopen('submission3.csv','w');
fprintf(fileID,'id,cancer\n');
for i = 1:length(id)
    fprintf(fileID,'%s,%d\n',id{i},submit(i));
end
fclose(fileID);