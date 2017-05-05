import matplotlib.pyplot as plt
import numpy as np

falsePos = np.load('rocCurveFiles/falsePos.npy')
falseNeg = np.load('rocCurveFiles/falseNeg.npy')
negThresholds = np.load('rocCurveFiles/negThresholds.npy')
posThresholds = np.load('rocCurveFiles/posThresholds.npy')
truePos = np.load('rocCurveFiles/truePos.npy')
trueNeg = np.load('rocCurveFiles/trueNeg.npy')

# plt.figure()
# lw = 2
# plt.plot(falsePos, truePos, color='darkorange',
#          lw=lw, label='ROC curve')
# plt.plot(falseNeg,trueNeg,color='black',lw=lw,label='Roc curve negative')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
print(posThresholds)
print(falsePos)
plt.figure()
lw = 2
plt.plot(posThresholds,falsePos, color='darkorange',
         lw=lw, label='false positive rate')
#plt.plot(truePos,posThresholds,color='black',lw=lw,label='true positive rate')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Threshold')
plt.ylabel('Rates')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()