s1 = load('rocInfo/NoduleDataROCPrep2.mat');
s2 = load('rocInfo/NoduleDataROCPrepEvenSplit.mat');

%%
s1 = load('rocInfo/PatientDataROCcurveBalancedSplit.mat');
s2 = load('rocInfo/PatientDataROCcurveUnbalancedSplit.mat');
s3 = load('rocInfo/PatientDataROCcurveOriginalSplit.mat');
s4 = load('rocInfo/PatientDataROCcurveUnbalancedOtherSplit.mat');
s5 = load('rocInfo/PatientDataROCcurveUnbalancedOtherSplit2.mat');

yhat2 = s1.yhatVal;
yhat2(yhat2<0)=0;
yhat2(yhat2>1)=1;

yhat3 = s2.yhatVal;
yhat3(yhat3<0)=0;
yhat3(yhat3>1)=1;

yhat4 = s3.yhatVal;
yhat4(yhat4<0)=0;
yhat4(yhat4>1)=1;

yhat5 = s4.yhatVal;
yhat5(yhat5<0)=0;
yhat5(yhat5>1)=1;

yhat6 = s5.yhatVal;
yhat6(yhat6<0)=0;
yhat6(yhat6>1)=1;

[XX,YY,Thresh,AUC] = perfcurve(s1.val_y,yhat2,1);
[XX2,YY2,Thresh2,AUC2] = perfcurve(s2.val_y,yhat3,1);
[XX3,YY3,Thresh3,AUC3] = perfcurve(s3.val_y,yhat4,1);
[XX4,YY4,Thresh4,AUC4] = perfcurve(s4.val_y,yhat5,1);
[XX5,YY5,Thresh5,AUC5] = perfcurve(s5.val_y,yhat6,1);
%[XX2,YY2,Thresh2,AUC2] = perfcurve(val_y,yhat2,1,'XCrit','fnr','YCrit','tnr');

baseVals = 0:0.1:1;

figure
hold on
plot(XX3,YY3,'b-')
plot(XX,YY,'g-')
plot(XX2,YY2,'r-')
plot(XX4,YY4,'k-')
plot(XX5,YY5,'C-')
plot(baseVals,baseVals,'b--')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%legend('1/3 Nodule, 2/3 No-Nodule Split','Even Split');
legend('Original Split: 0.22 Cancer, 0.78 No-Cancer',...
    '1/3 Cancer, 2/3 No-Cancer Split','Even Split', ...
    '2/3 Cancer, 1/3 No-Cancer Split',...
    '3/4 Cancer, 1/4 No-Cancer Split','Baseline ROC');
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