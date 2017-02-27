annoId1 = anno_seriesuid{1};
%%

annoInds = find(strcmp(annoId1,anno_seriesuid));
candInds = find(strcmp(annoId1,cand_seriesuid));
candInds2 = find(strcmp(annoId1,cand_seriesuid) & cand_class1==1);

%%

