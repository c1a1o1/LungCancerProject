curFeatData = zeros(64,64,100);

for i = 1:100
   curFeatData(:,:,i)=reshape(featureDataAlexNet(i,:),64,64); 
end