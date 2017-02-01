yValidByVolume = zeros(198,100,1000);
volInd = 1;
sliceInd=1;
for i = 1:19800
    yValidByVolume(volInd,sliceInd,:)=yValidPredAlex(i,:);
    sliceInd = sliceInd + 1;
    if(sliceInd > 100)
       sliceInd = 1;
       volInd = volInd + 1;
    end
end
%%

[probs,predictions] = max(yValidByVolume,[],2);
probs = reshape(probs,198,1000);
predictions = reshape(predictions,198,1000);