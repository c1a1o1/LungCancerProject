s1 = load('rocInfo/NoduleDataROCPrep2.mat');
s2 = load('rocInfo/NoduleDataROCPrepEvenSplit.mat');

%%
s1 = load('rocInfo/PatientDataROCcurveBalancedSplit.mat');
s2 = load('rocInfo/PatientDataROCcurveUnbalancedSplit.mat');

yhat2 = s1.yhatVal;
yhat2(yhat2<0)=0;
yhat2(yhat2>1)=1;

yhat3 = s2.yhatVal;
yhat3(yhat3<0)=0;
yhat3(yhat3>1)=1;

[XX,YY,Thresh,AUC] = perfcurve(s1.val_y,yhat2,1);
[XX2,YY2,Thresh2,AUC2] = perfcurve(s2.val_y,yhat3,1);
%[XX2,YY2,Thresh2,AUC2] = perfcurve(val_y,yhat2,1,'XCrit','fnr','YCrit','tnr');

baseVals = 0:0.1:1;

figure
hold on
plot(XX,YY,'g-')
plot(XX2,YY2,'r-')
plot(baseVals,baseVals,'b--')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%legend('1/3 Nodule, 2/3 No-Nodule Split','Even Split');
legend('1/3 Cancer, 2/3 No-Cancer Split','Even Split');
%title('ROC Curve on Nodule Prediction for Blocks');
title('ROC Curve on Cancer Prediction');
hold off
%{
figure
plot(Thresh,XX)
xlabel('Threshold')
ylabel('False Positive Rate')

figure
plot(Thresh,XX2)
xlabel('Threshold')
ylabel('False Negative Rate')

%}