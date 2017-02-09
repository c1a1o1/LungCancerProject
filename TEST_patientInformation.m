for i = 2:1398
   str1 = textdata{i,1};
   if(str1=='0ca943d821204ceb089510f836a367fd')
       i
       labelData(i)
   end
end
%%

%cancer patient
% slice 21 has something that seems to be a mass
resizedDCM = load('segFilesResized/resizedSegDCM_0ca943d821204ceb089510f836a367fd');
dcmVol = resizedDCM.resizedDCM;

addpath('imtool3D')
imtool3D(dcmVol)