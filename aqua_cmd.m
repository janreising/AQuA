%% setup
% -- preset 1: in vivo. 2: ex vivo. 3: GluSnFR
startup;  % initialize
load('random_Seed');
rng(s);

preset = 2;
input = 'C:\Users\janrei\Desktop\delete\test\';
channel=1;
total_channels=1;

[folder, name, ext] = fileparts(input);

%% determine preset
if exist('preset', 'var') == 0

    if contains(name, "10X") & contains(name, "ch2")
        preset = 1;
    elseif contains(name, "20X") & contains(name, "ch1")
        preset = 2;
    elseif contains(name, "20X") & contains(name, "ch2")
        preset = 3;
    else
        fprintf("Cannot choose preset automatically. Stopping the run!");
        return 
    end
end

%% options
opts = util.parseParam(preset,1);

%% read data
fprintf("\nLoading data ...\n");
tic;
if isfolder(input)
    [datOrg, opts] = tiff.load_from_folder(input, channel, total_channels, opts);
    
elseif strcmp(ext, '.tif') || strcmp(ext, '.tiff')
    [datOrg, opts] = tiff.load_from_tiff(input, opts);

else
    [datOrg,opts] = burst.prep1(p0,f0,[],opts);

end
toc; 

%% set file names
feature_path = strcat(opts.filePath,opts.fileName,'_FeatureTable.xlsx');
feature_path = feature_path{1}; % TODO why do I have to do this?
mat_path = strcat(opts.filePath,opts.fileName,'_AQuA.mat');
mat_path = mat_path{1};
h5_path = strcat(opts.filePath,opts.fileName,'.h5');
h5_path = h5_path{1};

%% detection
[dat,dF,arLst,lmLoc,opts,dL] = burst.actTop(datOrg,opts);  % foreground and seed detection
[svLst,~,riseX] = burst.spTop(dat,dF,lmLoc,[],opts);  % super voxel detection

[riseLst,datR,evtLst,seLst] = burst.evtTop(dat,dF,svLst,riseX,opts);  % events
[ftsLst,dffMat] = fea.getFeatureQuick(dat,evtLst,opts);

% fitler by significance level
mskx = ftsLst.curve.dffMaxZ>opts.zThr;
dffMatFilterZ = dffMat(mskx,:);
evtLstFilterZ = evtLst(mskx);
tBeginFilterZ = ftsLst.curve.tBegin(mskx);
riseLstFilterZ = riseLst(mskx);

% merging (glutamate)
if opts.ignoreMerge==0
    evtLstMerge = burst.mergeEvt(evtLstFilterZ,dffMatFilterZ,tBeginFilterZ,opts,[]);
else
    evtLstMerge = evtLstFilterZ;
end

% reconstruction (glutamate)
if opts.extendSV==0 || opts.ignoreMerge==0 || opts.extendEvtRe>0
    [riseLstE,datRE,evtLstE] = burst.evtTopEx(dat,dF,evtLstMerge,opts);
else
    riseLstE = riseLstFilterZ; datRE = datR; evtLstE = evtLstFilterZ;
end

% feature extraction
[ftsLstE,dffMatE,dMatE] = fea.getFeaturesTop(datOrg,evtLstE,opts);
ftsLstE = fea.getFeaturesPropTop(dat,datRE,evtLstE,ftsLstE,opts);

% update network features
sz = size(datOrg);
evtx1 = evtLstE;
ftsLstE.networkAll = [];
ftsLstE.network = [];
try
    ftsLstE.networkAll = fea.getEvtNetworkFeatures(evtLstE,sz);  % all filtered events
    ftsLstE.network = fea.getEvtNetworkFeatures(evtx1,sz);  % events inside cells only
catch
end

%% export to h5
if exist(h5_path, 'file') == 2
    fprintf("\nFile already exists. Choosing new name:\n");

    [folder, name, ext] = fileparts(h5_path);
    h5_path = [tempname(folder),'_',name,ext];

    disp(h5_path);
end

fprintf("\nStarting to save ... \n");
tic;
save_to_h5(h5_path, datOrg, '/datOrg');
save_to_h5(h5_path, opts, '/opts');
save_to_h5(h5_path, evtLstE, '/evtLstE');
save_to_h5(h5_path, ftsLstE, '/ftsLstE');
save_to_h5(h5_path, dffMatE, '/dffMatE');
save_to_h5(h5_path, dMatE, '/dMatE');
save_to_h5(h5_path, riseLstE, '/riseLstE');
save_to_h5(h5_path, datRE, '/datRE');
toc;

%% export table
fts = ftsLstE;
tb = readtable('userFeatures.csv','Delimiter',',');
if(isempty(ftsLstE.basic))
    nEvt = 0;
else
    nEvt = numel(ftsLstE.basic.area);
end
nFt = numel(tb.Name);
ftsTb = nan(nFt,nEvt);
ftsName = cell(nFt,1);
ftsCnt = 1;
dixx = ftsLstE.notes.propDirectionOrder;
lmkLst = [];

for ii=1:nFt
    cmdSel0 = tb.Script{ii};
    ftsName0 = tb.Name{ii};
    % if find landmark or direction
    if ~isempty(strfind(cmdSel0,'xxLmk')) %#ok<STREMP>
        for xxLmk=1:numel(lmkLst)
            try
                eval([cmdSel0,';']);
            catch
                fprintf('Feature "%s" not used\n',ftsName0)
                x = nan(nEvt,1);
            end
            ftsTb(ftsCnt,:) = reshape(x,1,[]);
            ftsName1 = [ftsName0,' - landmark ',num2str(xxLmk)];
            ftsName{ftsCnt} = ftsName1;
            ftsCnt = ftsCnt + 1;
        end
    elseif ~isempty(strfind(cmdSel0,'xxDi')) %#ok<STREMP>
        for xxDi=1:4
            try
                eval([cmdSel0,';']);
                ftsTb(ftsCnt,:) = reshape(x,1,[]);
            catch
                fprintf('Feature "%s" not used\n',ftsName0)
                ftsTb(ftsCnt,:) = nan;
            end            
            ftsName1 = [ftsName0,' - ',dixx{xxDi}];
            ftsName{ftsCnt} = ftsName1;
            ftsCnt = ftsCnt + 1;
        end
    else
        try
            eval([cmdSel0,';']);
            ftsTb(ftsCnt,:) = reshape(x,1,[]);            
        catch
            fprintf('Feature "%s" not used\n',ftsName0)
            ftsTb(ftsCnt,:) = nan;
        end
        ftsName{ftsCnt} = ftsName0;
        ftsCnt = ftsCnt + 1;
    end
end
featureTable = table(ftsTb,'RowNames',ftsName);
writetable(featureTable,feature_path,'WriteVariableNames',0,'WriteRowNames',1);

fprintf("Exported Feature table");
fprintf("\nProcessing finished.");
