sliceInfoFiles = dir('DOI_modSliceLoc/*.mat');
xmlFiles = dir('DOI_modXML/*.xml');

xmlFileInfoMap = getFileNameDict(xmlFiles,9,4);
sliceInfoFileMap = getFileNameDict(sliceInfoFiles,14,4);

allSliceInfoKeys = keys(sliceInfoFileMap);

xmlFilesForSliceInfo = values(xmlFileInfoMap,allSliceInfoKeys);
sliceInfoFilesUse = values(sliceInfoFileMap,allSliceInfoKeys);
malInfo = cell(1,length(xmlFilesForSliceInfo));
centerInfo = cell(1,length(xmlFilesForSliceInfo));
radInfo= cell(1,length(xmlFilesForSliceInfo));

for fileInd = 1:length(xmlFilesForSliceInfo)
%for fileInd = floor(rand*length(xmlFilesForSliceInfo)+1)
    fileInd
    
    clearvars -except fileInd xmlFilesForSliceInfo sliceInfoFilesUse centerInfo malInfo radInfo allSliceInfoKeys
    
    sliceLoc = load(strcat('DOI_modSliceLoc/',sliceInfoFilesUse{fileInd}));
    info = xml2struct(strcat('DOI_modXML/',xmlFilesForSliceInfo{fileInd}));

    zLocs = sliceLoc.locData;
    outputArray = zeros(512,512,length(zLocs));
    
    roiInfo = cell(0);
    roiInfoInd=1;
    
    if(~isfield(info.LidcReadMessage,'readingSession'))
       continue; 
    end

    numSessions = length(info.LidcReadMessage.readingSession);
    for sInd = 1:numSessions
        session = info.LidcReadMessage.readingSession{sInd};
        if(~isfield(session,'unblindedReadNodule'))
           continue; 
        end
        numNods = length(session.unblindedReadNodule);
        for nInd = 1:numNods
            if(numNods > 1)
                nodule = session.unblindedReadNodule{nInd};
            elseif(numNods==1)
                nodule = session.unblindedReadNodule;
            end

            %only used characterized nodules
            if(isfield(nodule,'characteristics'))
                
                if(~isfield(nodule.characteristics,'malignancy'))
                    continue;
                end
                
                malNum = str2double(nodule.characteristics.malignancy.Text);

                %care about ones where malignancy > 0
                %   malignancy number will be put into pixel
                numROI = length(nodule.roi);
                for rInd = 1:numROI
                    if(numROI>1)
                        curROI = nodule.roi{rInd};
                    else
                        curROI=nodule.roi;
                    end
                    
                    if(strcmpi(curROI.inclusion.Text,'FALSE'))
                        %exclude this ROI
                        continue;
                    end

                    curZ = str2double(curROI.imageZposition.Text);
                    zInd = find(zLocs==curZ);
                    numPts = length(curROI.edgeMap);
                    xPt = zeros(1,numPts);
                    yPt = zeros(1,numPts);
                    
                    if(numPts > 1)
                        for ii = 1:numPts
                           xPt(ii) = str2double(curROI.edgeMap{ii}.xCoord.Text); 
                           yPt(ii) = str2double(curROI.edgeMap{ii}.yCoord.Text); 
                        end
                        currentArea = polyarea(xPt,yPt);
                        currentBlob.center = [mean(xPt) mean(yPt) zInd];
                        currentBlob.radius = max(1,sqrt(currentArea)/pi);
                    else
                        xCoord = str2double(curROI.edgeMap.xCoord.Text);
                        yCoord = str2double(curROI.edgeMap.yCoord.Text);
                        currentBlob.center = [xCoord yCoord zInd];
                        currentBlob.radius = 1;
                    end

                    currentBlob.cancerLikelihood = malNum;
                    roiInfo{roiInfoInd}=currentBlob;
                    roiInfoInd = roiInfoInd+1;
                end


            end
        end

    end

    outputFile = strcat('DOI_modNoduleInfo/noduleInformation_',allSliceInfoKeys{fileInd},'.mat');
    
    numROIfound = roiInfoInd-1;
    malignancies = zeros(numROIfound,1);
   nodCenters = zeros(numROIfound,3);
   nodRadii = zeros(numROIfound,1);
    if(numROIfound > 0)
       for ii = 1:numROIfound
          malignancies(ii)=roiInfo{ii}.cancerLikelihood;
          nodCenters(ii,:)=roiInfo{ii}.center;
          nodRadii(ii,:)=roiInfo{ii}.radius;
       end
    end
    save(outputFile,'roiInfo','malignancies','nodCenters','nodRadii');
    
    centerInfo{fileInd}=nodCenters;
    radInfo{fileInd}=nodRadii;
    malInfo{fileInd}=malignancies;
    
    
end

save('ALLFILES_infoFromXML.mat','centerInfo','radInfo','malInfo','allSliceInfoKeys');

%{
malArray = [];
for ii = 1:length(malInfo)
    currentArr = malInfo{ii};
    if(~isempty(currentArr))
       malArray = [malArray currentArr(:)'];
    end
end
%}



