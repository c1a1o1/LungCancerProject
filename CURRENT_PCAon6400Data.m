
%[coeff1,score1,latent,tsquared,explained,mu1] = pca(allXdata);
%save('allXDataAndAllPCAInfo.mat','allXdata','coeff1','score1','latent','tsquared','explained','mu1');
load('allXDataAndAllPCAInfo.mat');
load('xData6400autoencoded.mat');
reconAutoEn = [trainTestRecon;validationRecon];
reconAutoDiff = reconAutoEn-allXdata;
reconAutoAvg = mean(abs(reconAutoDiff(:)));

numDim=512;

numTotalDim = size(score1,2);
reconError = zeros(1,numTotalDim);

for kk = 1:numTotalDim
    kk
    score2 = score1(:,1:kk); coeff2 = coeff1(:,1:kk);
    recon2 = score2*coeff2'+repmat(mu1,size(allXdata,1),1);
    recon2diff = recon2-allXdata;
    reconError(kk) = mean(abs(recon2diff(:)));
end

save('pcaReconTest.mat','reconError','reconAutoAvg');
%%

xOutput500dim = score1(:,1:500);
save('newXfromPCA.mat','xOutput500dim');

