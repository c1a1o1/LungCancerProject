lineInfo = cell(1,3);

%lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C1_D1.txt','\n');
%lineInfo{2} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C2_D1.txt','\n');
%lineInfo{3} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C3_D1.txt','\n');

lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C1_D1.txt','\n');
lineInfo{2} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C2_D1.txt','\n');
lineInfo{3} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C3_D1.txt','\n');

validationLoss = cell(1,3);

for kk = 1:3
    
    lines = lineInfo{kk};
    
    validationLogLoss = [];
    for linNum = 1:length(lines)
       curStr = lines{linNum};
       if(~isempty(strfind(curStr,'validation_0-logloss:')))
          %fprintf(strcat(curStr,'\n'))
          pieces = strsplit(curStr,':');
          validationLogLoss = [validationLogLoss str2double(pieces{2})];
       end

    end
    validationLogLoss = validationLogLoss(1:end-1);
    validationLoss{kk} = validationLogLoss;
end

figure
hold on
plot(validationLoss{3},'b-')
plot(validationLoss{1},'r-')
plot(validationLoss{2},'g-')
legend('Mean only','Max,Mean concat','Max,Min,Mean concat');
xlabel('Num Rounds of XGBoost');
ylabel('Validation Log Loss');
hold off
