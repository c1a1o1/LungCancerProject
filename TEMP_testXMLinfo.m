xmlFiles = dir('DOI_modXML/*.xml');
sliceInfoFiles = dir('DOI_modMatlab/*.mat');
dcmInfoFiles = dir('DOI_dcmInfo/*.mat');

xmlFileInfoMap = getFileNameDict(xmlFiles,9,4);
sliceInfoFileMap = getFileNameDict(sliceInfoFiles,14,8);
dcmFileInfoMap = getFileNameDict(dcmInfoFiles,13,4);

sliceInfoIDs = keys(sliceInfoFileMap);


for ind = 1:length(sliceInfoIDs)
    fprintf(strcat('Processing slice ',num2str(ind),' of ',num2str(length(sliceInfoIDs)),'\n'));
    sliceInfoFileName = values(sliceInfoFileMap,sliceInfoIDs(ind));
    xmlFileName = values(xmlFileInfoMap,sliceInfoIDs(ind));
    dcmInfoFileNm = values(dcmFileInfoMap,sliceInfoIDs(ind));

    info = xml2struct(strcat('DOI_modXML/',xmlFileName{1}));
    load(strcat('DOI_dcmInfo/',dcmInfoFileNm{1}));
    Xspacing = 1; Yspacing = 1;
    if(isfield(dcmArray{1},'PixelSpacing'))
        spacing = dcmArray{1}.PixelSpacing;
        Xspacing = spacing(1);
        Yspacing = spacing(2);
    end
    
    %sliceLoc = load(strcat('DOI_modMatlab/',sliceInfoFileName{1}));
    %zLocs = sliceLoc.sliceZ;
    %outputArray = zeros(512,512,length(zLocs));

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

                        %curZ = str2double(curROI.imageZposition.Text);
                        %zInd = find(zLocs==curZ);
                        numPts = length(curROI.edgeMap);
                        xPt = zeros(1,numPts);
                        yPt = zeros(1,numPts);

                        for ii = 1:numPts
                           xPt(ii) = str2double(curROI.edgeMap{ii}.xCoord.Text); 
                           yPt(ii) = str2double(curROI.edgeMap{ii}.yCoord.Text); 
                        end
                        
                        vv1 = max(xPt);
                        vv2 = max(yPt);
                        
                        %{
                        This is a test to see if it uses Pixel Coordinates
                            or X,Y Coordinates in mm. 
                        If it uses x,y in mm coords, then 512*Xspacing
                            would be max read in x direction. Similarly, 
                            512*Yspacing would be max in that direction. 
                        It is specified that (0,0) is the upper left
                            coordinate in x,y space that they use. 
                        If it passes this test, then pixel coordinates must
                            be used as x,y coordinates would not work
                        
                        VERDICT: It does pass the test, pixel coordinates
                        are the ones used
                        %}
                        if(vv1 > 512*Xspacing || vv2 > 512*Yspacing )
                            fprintf('Uses pixel coords!!\n');
                        end
                        
                        %{
                        This double checks the x-y coordinates
                        %}
                        if(vv1 > 512 || vv2 > 512 )
                            fprintf('Bad Pixel Coords :( \n');
                        end

                        %makes binary matrix, 1 means inside polygon. 0 otherwise
                        inMatrix = inpolygon(XX,YY,xPt,yPt);

                        %outputArray(:,:,zInd)=double(inMatrix);
                    end
                end


            end
        end

    end
    
end
