load('NoduleDataROCPrep2.mat')

yhat2 = yhatVal;
yhat2(yhat2<0)=0;
yhat2(yhat2>1)=1;

[XX,YY,Thresh,AUC] = perfcurve(val_y,yhat2,1);
[XX2,YY2,Thresh2,AUC2] = perfcurve(val_y,yhat2,1,'XCrit','fnr','YCrit','tnr');

figure
plot(XX,YY)
xlabel('False Positive Rate')
ylabel('True Positive Rate')

figure
plot(Thresh,XX)
xlabel('Threshold')
ylabel('False Positive Rate')

figure
plot(Thresh,XX2)
xlabel('Threshold')
ylabel('False Negative Rate')