lines = importdata('resNetTo3DConvOutput1.txt','\n');
trainingLoss = [];
validationLoss = [];
trainingAccu = [];
validationAccu = [];
for linNum = 1:length(lines)
   curStr = lines{linNum};
   if(~isempty(strfind(curStr,'val_loss')) || ~isempty(strfind(curStr,'Epoch')))
      fprintf(strcat(curStr,'\n'))
   end
   if(~isempty(strfind(curStr,'val_loss')))
       pieces = strsplit(curStr,' ');
       for pieceInd= 1:length(pieces)
           current = pieces{pieceInd};
          if(strcmp(current,'loss:'))
              curNum = str2double(pieces{pieceInd+1});
             trainingLoss = [trainingLoss curNum];
          end
          if(strcmp(current,'val_loss:'))
              curNum = str2double(pieces{pieceInd+1});
              validationLoss = [validationLoss curNum];
          end
          if(strcmp(current,'acc:'))
              curNum = str2double(pieces{pieceInd+1});
             trainingAccu = [trainingAccu curNum];
          end
          if(strcmp(current,'val_acc:'))
              curNum = str2double(pieces{pieceInd+1});
              validationAccu = [validationAccu curNum];
          end
       end
   end
end
%%
trainLoss512 = trainingLoss(1:15);
validLoss512 = validationLoss(1:15);

trainLoss2048 = trainingLoss(16:30);
validLoss2048 = validationLoss(16:30);

trainAccu512 = trainingAccu(1:15);
validAccu512 = validationAccu(1:15);

trainAccu2048 = trainingAccu(16:30);
validAccu2048 = validationAccu(16:30);

figure
hold on
plot(trainLoss512,'r--')
plot(validLoss512,'r-','LineWidth',2)
plot(trainLoss2048,'k--')
plot(validLoss2048,'k-','LineWidth',2)
xlabel('# Epoch');
ylabel('Log Loss');
legend('Training Loss for (A)->(B1)->(C1)','Validation Loss for (A)->(B1)->(C1)',...
    'Training Loss for (A)->(B2)->(C1)','Validation Loss for (A)->(B2)->(C1)');
hold off

figure
hold on
plot(trainAccu512,'r--')
plot(validAccu512,'r-','LineWidth',2)
plot(trainAccu2048,'k--')
plot(validAccu2048,'k-','LineWidth',2)
xlabel('# Epoch');
ylabel('Log Loss');
legend('Training Accu for 512x7x7','Validation Accu for 512x7x7',...
    'Training Accu for 2048x7x7','Validation Accu for 2048x7x7');
hold off


