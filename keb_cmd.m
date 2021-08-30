%% setup
% -- preset 1: in vivo. 2: ex vivo. 3: GluSnFR
startup;  % initialize
load('random_Seed');
rng(s);

%preset = 2;
%file = "/media/carmichael/1TB/delete/1-40X-loc1.zip.h5";
%run keb_cmd.m

%% save path
[folder, name, ext] = fileparts(file);
p0 = strcat(folder,filesep);
f0 = strcat(name,ext);
path0 = [p0,name,filesep];

feature_path = strcat(p0,name,'_FeatureTable.xlsx');
h5_path = strcat(p0,name,'_AQuA.h5');

%% options
%opts = util.parseParam(preset,1);


opts = {};
opts.minSize = 10;  % minimum size % 50
opts.smoXY = 0.1; % spatial smoothing level % 0.5
opts.thrARScl = 3; % active voxel threshold % 1.5

opts.thrTWScl = 10; % temporal cut threshold % 1.5
opts.thrExtZ = 3; % Seed growing threshold %1.5

opts.cDelay = 1; % Slowest propagation
opts.cRise = 1; % rising phase uncertainty
opts.gtwSmo = 2; % GTW smoothness term
opts.maxStp = 11; % GTW windows size
opts.zThr = 6; % Z score threshold for events

opts.ignoreMerge = 0; % 0 % Ignore merging step
opts.mergeEventDiscon = 10; % 10 % Maximum merging distance
opts.mergeEventCorr = 1; % 1 % Minimum merging correlation
opts.mergeEventMaxTimeDif = 2; % 2 % Maximum merging time difference


opts.regMaskGap = 1; % Remove pixels close to image boundary
opts.usePG = 1; % Poisson noise model
opts.cut = 200; % Frames per segment
opts.movAvgWin = 25; % Baseline window
opts.extendSV = 1; % Extend super voxels temporally
opts.legacyModeActRun = 1; % Older code for active voxels
opts.getTimeWindowExt = 50; % Time window detection range
opts.seedNeib = 1; % Pixels for window detection
opts.seedRemoveNeib = 2; % Remove seeds
opts.thrSvSig = 4; % Super voxel significance
opts.gapExt = 5; % Check more time
opts.superEventdensityFirst = 1; % Super events prefer larger
opts.gtwGapSeedRatio = 4; % Area ratio to find seed curve
opts.gtwGapSeedMin = 5; % Area to find seed curve
opts.cOver = 0.2; % Spatial overlap threshold
opts.minShow1 = 0.2; % Event show threshold on raw data
opts.minShowEvtGUI = 0; % GUI event boundary threshold
opts.ignoreTau = 0; % Ignore decay tau calculation
opts.correctTrend = 1; % Correct baseline trend
opts.extendEvtRe = 1; % Extend event temporally after merging
opts.propthrmin = 0.2; % Propagation threshold minimum
opts.propthrstep = 0.1; % Propagation threshold step
opts.propthrmax = 0.8; % Propagation threshold maximum

opts.frameRate = 0.125; % Frame rate
opts.spatialRes = 0.55; % Spatial resolution
opts.varEst = 0.02; % Estimated noise variance
opts.fgFluo = 0; % Foreground threshold
opts.bgFluo = 0; % Background threshold
opts.northx = 0; % X cooridante for north vector
opts.northy = 1; % Y cooridante for north vector
opts.skipSteps = 0; % Skip step2 and 3

[datOrg,opts] = burst.prep1(p0,f0,[],opts);  % read data

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
    h5_path = strcat(tempname(folder),'_',name,ext);

    disp(h5_path);
end

fprintf("\nStarting to save ... \n");
tic;
save_to_h5(h5_path, datOrg, '/res/datOrg');
save_to_h5(h5_path, opts, '/res/opts');
save_to_h5(h5_path, evtLstE, '/res/evt');
save_to_h5(h5_path, ftsLstE, '/res/fts');
save_to_h5(h5_path, dffMatE, '/res/dffMat');
save_to_h5(h5_path, dMatE, '/res/dMat');
save_to_h5(h5_path, riseLstE, '/res/rise');
save_to_h5(h5_path, datRE, '/res/datR');
toc;


fprintf("\nProcessing finished.");
