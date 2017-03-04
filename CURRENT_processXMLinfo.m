%file1 = dir('randFiles/*.xml');
%file2 = dir('randFiles/*.mat');

dcmInfoFiles = dir('DOI_dcmInfo/*.mat');
xmlFiles = dir('DOI_modXML/*.xml');
%%

filex = dcmInfoFiles(1).name;
filex(14:end-4); %gets the patient ID 

filey = xmlFiles(1).name;
filey(9:end-4); %gets the patient ID

dcmFileNames = cell(1,length(dcmInfoFiles));
dcmFilePatIDs = cell(1,length(dcmInfoFiles));
for ii = 1:length(dcmInfoFiles)
    currentFileName = dcmInfoFiles(ii).name;
    currentPatId = currentFileName(14:end-4);
    dcmFileNames{ii} = currentFileName;
    dcmFilePatIDs{ii} = currentPatId;
end
dcmFileInfoMap = containers.Map(dcmFilePatIDs,dcmFileNames);

xmlFileNames = cell(1,length(xmlFiles));
xmlFilePatIDs = cell(1,length(xmlFiles));
for ii = 1:length(xmlFiles)
    currentFileName = xmlFiles(ii).name;
    currentPatId = currentFileName(9:end-4);
    xmlFileNames{ii} = currentFileName;
    xmlFilePatIDs{ii} = currentPatId;
end
xmlFileInfoMap = containers.Map(xmlFilePatIDs,xmlFileNames);
%%

allDCMkeys = keys(dcmFileInfoMap);
xmlFilesForDCM = values(xmlFileInfoMap,allDCMkeys);
allXMLkeys = keys(xmlFileInfoMap);
dcmFilesForXML = values(dcmFileInfoMap,allXMLkeys);


%%
sliceLoc = load(strcat('randFiles/',file2.name));
info = xml2struct(strcat('randFiles/',file1.name));

zLocs = sliceLoc.sliceZ;
outputArray = zeros(512,512,length(zLocs));

%makes grid of pts
[XX,YY]=meshgrid(1:512,1:512);


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
            malNum = str2double(nodule.characteristics.malignancy.Text);
            
            %care about ones where malignancy > 1
            if(malNum>1)
                numROI = length(nodule.roi);
                for rInd = 1:numROI
                    curROI = nodule.roi{rInd};
                    
                    curZ = str2double(curROI.imageZposition.Text);
                    zInd = find(zLocs==curZ);
                    numPts = length(curROI.edgeMap);
                    xPt = zeros(1,numPts);
                    yPt = zeros(1,numPts);

                    for ii = 1:numPts
                       xPt(ii) = str2double(curROI.edgeMap{ii}.xCoord.Text); 
                       yPt(ii) = str2double(curROI.edgeMap{ii}.yCoord.Text); 
                    end

                    %makes binary matrix, 1 means inside polygon. 0 otherwise
                    inMatrix = inpolygon(XX,YY,xPt,yPt);
                    
                    outputArray(:,:,zInd)=double(inMatrix);
                end
            end
            
            
        end
    end
    
end
