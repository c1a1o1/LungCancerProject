lineInfo = cell(1,3);

%figure 1
%lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C1_D1.txt','\n');
%lineInfo{2} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C2_D1.txt','\n');
%lineInfo{3} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C3_D1.txt','\n');

%figure 2
%lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C1_D1.txt','\n');
%lineInfo{2} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C2_D1.txt','\n');
%lineInfo{3} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C3_D1.txt','\n');

%figure 3
%lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C1_D1.txt','\n');
%lineInfo{2} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C2_D1.txt','\n');
%lineInfo{3} = importdata('progOutput/resNetFeatsToXGBoost_A1_BC4_D1.txt','\n');

%figure 4
%lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B1_C3_D1.txt','\n');
%lineInfo{2} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C3_D1.txt','\n');
%lineInfo{3} = importdata('progOutput/resNetFeatsToXGBoost_A1_BC5_D1.txt','\n');

%figure 5
lineInfo{1} = importdata('progOutput/resNetFeatsToXGBoost_A1_B2_C2_D1.txt','\n');
lineInfo{2} = importdata('progOutput/meeting_2_23_xgBoostTuning_maxDepth10.txt','\n');
lineInfo{3} = importdata('progOutput/meeting_2_23_xgBoost_maxDepth5.txt','\n');


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
plot(validationLoss{1},'r-')
plot(validationLoss{2},'g-')
plot(validationLoss{3},'b-')
legend('Max Depth 30','Max Depth 10','Max Depth 5');
%legend('A1->B1->C3->D1','A1->B2->C3->D1','A1->BC5->D1');
%legend('Mean only','Max,Mean concat','Max,Min,Mean concat');
xlabel('Num Rounds of XGBoost');
ylabel('Validation Log Loss');
hold off
