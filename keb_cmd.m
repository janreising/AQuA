%% setup
% -- preset 1: in vivo. 2: ex vivo. 3: GluSnFR
startup;  % initialize
load('random_Seed');
rng(s);

%preset = 2;
%file = "/home/carmichael/Downloads/2-40X-loc1.cnmfe.h5";

%% save path
[folder, name, ext] = fileparts(file);
p0 = strcat(folder,filesep);
f0 = strcat(name,ext);
path0 = [p0,name,filesep];

feature_path = strcat(p0,name,'_FeatureTable.xlsx');
h5path = strcat(p0,name,'_AQuA.h5');

%% determine preset

%{
if contains(name, "10X") & contains(name, "ch2")
    preset = 1;
elseif contains(name, "20X") & contains(name, "ch1")
    preset = 2;
elseif contains(name, "20X") & contains(name, "ch2")
    preset = 3;
else
    %fprintf("Cannot choose preset automatically. Stopping the run!");
    %return
    preset = 2;
    fprintf("Cannot choose automatically. Choosing 2");
end
%}

%% options
opts = util.parseParam(preset,1);

% opts.smoXY = 1;
% opts.thrARScl = 2;
% opts.movAvgWin = 15;
% opts.minSize = 8;
% opts.regMaskGap = 0;
% opts.thrTWScl = 5;
% opts.thrExtZ = 0.5;
% opts.extendSV = 1;
% opts.cRise = 1;
% opts.cDelay = 2;
% opts.zThr = 3;
% opts.getTimeWindowExt = 10000;
% opts.seedNeib = 5;
% opts.seedRemoveNeib = 5;
% opts.thrSvSig = 1;
% opts.extendEvtRe = 0;

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
    h5_path = [tempname(folder),'_',name,ext];

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
