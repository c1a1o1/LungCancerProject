prismValid = ones(512,512,124);

numPt=4500;
pts = floor(rand(numPt,3).*repmat([512-64 512-64 124-64],numPt,1)+36);
finalPts = [];
for ii = 1:size(pts,1)
    curPt = pts(ii,:);
    if(prismValid(curPt(1),curPt(2),curPt(3))>0)
        finalPts = [finalPts;curPt];
        prismValid(curPt(1)-32:curPt(1)+32,curPt(2)-32:curPt(2)+32,curPt(3)-32:curPt(3)+32)=0;
    end
end

%%

%load('tempTest/binaryNoduleMatrix_1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mat');
%load('tempTest/HUarrayResizeInfo_1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mat');

%load('tempTest/binaryNoduleMatrix_1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mat');
%load('tempTest/HUarrayResizeInfo_1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mat');

load('tempTest/binaryNoduleMatrix_1.3.6.1.4.1.14519.5.2.1.6279.6001.102962801900681595502684962582.mat');
load('tempTest/HUarrayResizeInfo_1.3.6.1.4.1.14519.5.2.1.6279.6001.102962801900681595502684962582.mat');


displayBinaryArray = zeros(512,512,length(finalOutputSparse));
for ii = 1:length(finalOutputSparse)
   displayBinaryArray(:,:,ii)=finalOutputSparse{ii}; 
end

displayPixelArray = pixelArray.*displayBinaryArray;
imtool3D(displayPixelArray)
imtool3D(pixelArray)