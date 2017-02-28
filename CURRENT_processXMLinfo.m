file = 'randFiles/072.xml';
info = xml2struct(file);
%%
roi1 = info.LidcReadMessage.readingSession{2}.unblindedReadNodule{1}.roi{1};

numPts = length(roi1.edgeMap);
xPt = zeros(1,numPts);
yPt = zeros(1,numPts);

for ii = 1:numPts
   xPt(ii) = str2double(roi1.edgeMap{ii}.xCoord.Text); 
   yPt(ii) = str2double(roi1.edgeMap{ii}.yCoord.Text); 
end

%makes grid of pts
[XX,YY]=meshgrid(1:512,1:512);

%makes binary matrix, 1 means inside polygon. 0 otherwise
in = inpolygon(XX,YY,xPt,yPt);

%displays it
imagesc(double(in))