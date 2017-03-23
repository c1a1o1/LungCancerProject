s1 = load('rocInfo/NoduleDataROCPrep2.mat');
s2 = load('rocInfo/NoduleDataROCPrepEvenSplit.mat');
s3 = load('rocInfo/NoduleDataROCPrepUnbalancedOtherSplit.mat');

yhat2 = s1.yhatVal;
yhat2(yhat2<0)=0;
yhat2(yhat2>1)=1;

yhat3 = s2.yhatVal;
yhat3(yhat3<0)=0;
yhat3(yhat3>1)=1;

yhat4 = s3.yhatVal;
yhat4(yhat4<0)=0;
yhat4(yhat4>1)=1;
%{
yhat5 = s4.yhatVal;
yhat5(yhat5<0)=0;
yhat5(yhat5>1)=1;

yhat6 = s5.yhatVal;
yhat6(yhat6<0)=0;
yhat6(yhat6>1)=1;
%}
[XX,YY,Thresh,AUC] = perfcurve(s1.val_y,yhat2,1);
[XX2,YY2,Thresh2,AUC2] = perfcurve(s2.val_y,yhat3,1);
[XX3,YY3,Thresh3,AUC3] = perfcurve(s3.val_y,yhat4,1);

baseVals = 0:0.1:1;

figure
hold on
plot(XX,YY,'g-')
plot(XX2,YY2,'r-')
plot(XX3,YY3,'k-')
plot(baseVals,baseVals,'b--')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
legend('1/3 Nodule, 2/3 No-Nodule Split','Even Split','2/3 Nodule, 1/3 No-Nodule Split','Baseline');
title('ROC Curve on Nodule Detection');
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