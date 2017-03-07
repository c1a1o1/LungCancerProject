%file1 = dir('randFiles/*.xml');
%file2 = dir('randFiles/*.mat');

%dcmInfoFiles = dir('DOI_dcmInfo/*.mat');
sliceInfoFiles = dir('DOI_modSliceLoc/*.mat');
xmlFiles = dir('DOI_modXML/*.xml');

%{
filex = dcmInfoFiles(1).name;
filex(13:end-4); %gets the patient ID 

filey = xmlFiles(1).name;
filey(9:end-4); %gets the patient ID
%}

xmlFileInfoMap = getFileNameDict(xmlFiles,9,4);
sliceInfoFileMap = getFileNameDict(sliceInfoFiles,14,4);

allSliceInfoKeys = keys(sliceInfoFileMap);

xmlFilesForSliceInfo = values(xmlFileInfoMap,allSliceInfoKeys);
sliceInfoFilesUse = values(sliceInfoFileMap,allSliceInfoKeys);

%makes grid of pts
[YY,XX]=meshgrid(1:512,1:512);

for fileInd = 215:length(xmlFilesForSliceInfo)
    fileInd
    
    clearvars -except fileInd xmlFilesForSliceInfo sliceInfoFilesUse XX YY allSliceInfoKeys
    
    sliceLoc = load(strcat('DOI_modSliceLoc/',sliceInfoFilesUse{fileInd}));
    info = xml2struct(strcat('DOI_modXML/',xmlFilesForSliceInfo{fileInd}));

    zLocs = sliceLoc.locData;
    outputArray = zeros(512,512,length(zLocs));
    
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

                    curZ = str2double(curROI.imageZposition.Text);
                    zInd = find(zLocs==curZ);
                    numPts = length(curROI.edgeMap);
                    xPt = zeros(1,numPts);
                    yPt = zeros(1,numPts);
                    
                    if(numPts > 1)
                        for ii = 1:numPts
                           xPt(ii) = str2double(curROI.edgeMap{ii}.xCoord.Text); 
                           yPt(ii) = str2double(curROI.edgeMap{ii}.yCoord.Text); 
                           
                           %makes binary matrix, 1 means inside polygon. 0 otherwise
                            inMatrix = inpolygon(XX,YY,xPt,yPt);
                        end
                    else
                        xCoord = str2double(curROI.edgeMap.xCoord.Text);
                        yCoord = str2double(curROI.edgeMap.yCoord.Text);
                        inMatrix = false(512,512);
                        inMatrix(yCoord-1:yCoord+1,xCoord-1:xCoord+1)=true;
                    end

                    

                    currentSlice = outputArray(:,:,zInd);

                    if(strcmpi(curROI.inclusion.Text,'FALSE'))
                        %exclude the ROI pixels
                        inMatrix2 = double(inMatrix).*(-0.5);
                    else
                        inMatrix2 = double(inMatrix).*malNum;

                    end

                    newSlice = currentSlice+inMatrix2;
                    outputArray(:,:,zInd)=newSlice;
                end


            end
        end

    end

    finalOutput = double(outputArray>0);
    finalOutputSparse = cell(1,size(finalOutput,3));
    for spInd = 1:size(finalOutput,3)
       finalOutputSparse{spInd} = sparse(finalOutput(:,:,spInd)); 
    end

    outputFile = strcat('DOI_modNodule/binaryNoduleMatrix_',allSliceInfoKeys{fileInd},'.mat');
    save(outputFile,'finalOutputSparse');
    
end
