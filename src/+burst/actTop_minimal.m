function [dat,dF,arLst,lmLoc,opts,dActVox] = actTop(dat,opts,evtSpatialMask,ff)

    [H,W,T] = size(dat);
    msk000 = var(dat,0,3)>1e-8;
    evtSpatialMask = msk000;

    % smooth the data
    if opts.smoXY>0
        for tt=1:size(dat,3)
            dat(:,:,tt) = imgaussfilt(dat(:,:,tt),opts.smoXY);
        end
    end

    % noise and threshold, get active voxels
    [arLst,dActVox] = burst.getARSim(datOrg,opts,evtSpatialMask,opts.smoXY,opts.thrARScl,opts.minSize);

    % seeds
    fsz = [1 1 0.5];  % smoothing for seed detection
    lmLoc = burst.getLmAll(dat,arLst,dActVox,fsz);

end


function [arLst,dARAll] = getARSim(dat,opts,evtSpatialMask,smoMax,thrMin,minSize)

    % learn noise correlation
    T1 = min(size(dat,3),100);
    datZ = zscore(dat(:,:,1:T1),0,3);
    rhox = mean(datZ(:,1:end-1,:).*datZ(:,2:end,:),3);
    rhoy = mean(datZ(1:end-1,:,:).*datZ(2:end,:,:),3);
    rhoxM = nanmedian(rhox(:));
    rhoyM = nanmedian(rhoy(:));

    rr = load('smoCorr.mat');
    [~,ix] = min(abs(rhoxM-rr.cx));
    [~,iy] = min(abs(rhoyM-rr.cy));
    smo0 = rr.sVec(max(ix,iy));

    dSim = randn(opts.sz(1),opts.sz(2),200)*0.2;
    dSim = imgaussfilt(dSim,[smo0 smo0]);

    rto = size(dat,3)/size(dSim,3);

    % simulation
    smoVec = smoMax;
    thrVec = thrMin+3:-1:thrMin;

    dARAll = zeros(opts.sz);
    for ii=1:numel(smoVec)
        fprintf('Smo %d ==== \n',ii);
        opts.smoXY = smoVec(ii);

        [~,dFSim,sSim] = burst.arSimPrep(dSim,opts);
        [~,dFReal,sReal] = burst.arSimPrep(dat,opts);
        for jj=1:numel(thrVec)
            dAR = zeros(opts.sz);
            fprintf('Thr %d \n',jj);

            % null
            tmpSim = dFSim>thrVec(jj)*sSim;
            szFreqNull = zeros(1,opts.sz(1)*opts.sz(2));
            for tt=1:size(dSim,3)
                tmp00 = tmpSim(:,:,tt).*evtSpatialMask;
                cc = bwconncomp(tmp00);
                ccSz = cellfun(@numel,cc.PixelIdxList);
                for mm=1:numel(ccSz)
                    szFreqNull(ccSz(mm)) = szFreqNull(ccSz(mm))+1;
                end
            end
            szFreqNull = szFreqNull*rto;

            % observation
            tmpReal = dFReal>thrVec(jj)*sReal;
            szFreqObs = zeros(1,opts.sz(1)*opts.sz(2));
            for tt=1:size(dat,3)
                tmp00 = tmpReal(:,:,tt).*evtSpatialMask;
                cc = bwconncomp(tmp00);
                ccSz = cellfun(@numel,cc.PixelIdxList);
                for mm=1:numel(ccSz)
                    szFreqObs(ccSz(mm)) = szFreqObs(ccSz(mm))+1;
                end
            end

            % false positive control
            suc = 0;
            szThr = 0;
            for mm=1:opts.sz(1)*opts.sz(2)
                if sum(szFreqObs(mm:end))==0
                    break
                end
                fpr = sum(szFreqNull(mm:end))/sum(szFreqObs(mm:end));
                if fpr<0.01
                    suc = 1;
                    szThr = ceil(mm*1.2);
                    break
                end
            end
            szThr = max(szThr,minSize);

            % apply to data
            if suc>0
                e00 = round(smoVec(ii)/2);
                for tt=1:size(dat,3)
                    tmp0 = tmpReal(:,:,tt).*evtSpatialMask;
                    tmp0 = bwareaopen(tmp0,szThr);
                    if e00>0
                        tmp0 = imerode(tmp0,strel('square',e00));
                    end
                    dAR(:,:,tt) = dAR(:,:,tt) + tmp0;
                end
            end
            %zzshow(dAR);
            %keyboard
            dARAll = dARAll + dAR;
        end
    end

    dARAll = dARAll>0;
    arLst = bwconncomp(dARAll);
    arLst = arLst.PixelIdxList;

end

function [dat,dF,stdEst] = arSimPrep(dat,opts)

    mskSig = var(dat,0,3)>1e-8;
    dat = dat + randn(size(dat))*1e-6;

    if opts.smoXY>0
        for tt=1:size(dat,3)
            dat(:,:,tt) = imgaussfilt(dat(:,:,tt),opts.smoXY);
        end
    end

    % noise estimation
    xx = (dat(:,:,2:end)-dat(:,:,1:end-1)).^2;
    stdMap = sqrt(median(xx,3)/0.9133);
    stdMap(~mskSig) = nan;
    stdEst = double(nanmedian(stdMap(:)));

    dF = burst.getDfBlk(dat,mskSig,opts.cut,opts.movAvgWin,stdEst);

end

function [lmLoc,lmVal] = getLmAll(dat,arLst,dActVox,fsz)
% detect all local maximums in a movie

gaph = 3;
gapt = 3;
[H,W,T] = size(dat);
% arLst = label2idx(dL);
nAR = numel(arLst);

% detect in active region only
lmAll = zeros(size(dat),'logical');
for ii=1:nAR
    if mod(ii,1000)==0; fprintf('%d/%d\n',ii,nAR); end
    kk = ii;
    pix0 = arLst{kk};
    if isempty(pix0)
        continue
    end
    [ih,iw,it] = ind2sub([H,W,T],pix0);

    rgH = max(min(ih)-gaph,1):min(max(ih)+gaph,H);
    rgW = max(min(iw)-gaph,1):min(max(iw)+gaph,W);
    rgT = max(min(it)-gapt,1):min(max(it)+gapt,T);

    dInST = dat(rgH,rgW,rgT);
    ih1 = ih - min(rgH) + 1;
    iw1 = iw - min(rgW) + 1;
    it1 = it - min(rgT) + 1;
    pix0a = sub2ind(size(dInST),ih1,iw1,it1);
    mskST = false(size(dInST));
    mskST(pix0a) = true;
    mskSTSeed = dActVox(rgH,rgW,rgT);
    mskSTSeed = mskST & mskSTSeed;
    [~,~,lm3Idx] = burst.getLocalMax3D(dInST,mskSTSeed,mskST,fsz);
    lmAll(rgH,rgW,rgT) = max(lmAll(rgH,rgW,rgT),lm3Idx>0);
end
lmLoc = find(lmAll>0);
lmVal = dat(lmLoc);
[lmVal,ix] = sort(lmVal,'descend');
lmLoc = lmLoc(ix);

end

function dF = getDfBlk(datIn,evtSpatialMask,cut,movAvgWin,stdEst)

    if ~exist('stdEst','var') || isempty(stdEst)
        xx = (datIn(:,:,2:end)-datIn(:,:,1:end-1)).^2;
        stdMap = sqrt(median(xx,3)/0.9133);
        stdMap(evtSpatialMask==0) = nan;
        stdEst = double(nanmedian(stdMap(:)));
    end

    [H,W,T] = size(datIn);
    dF = zeros(H,W,T,'single');

    xx = randn(10000,cut)*stdEst;
    xxMA = movmean(xx,movAvgWin,2);
    xxMin = min(xxMA,[],2);
    xBias = nanmean(xxMin(:));

    nBlk = max(floor(T/cut),1);
    for ii=1:nBlk
        t0 = (ii-1)*cut+1;
        if ii==nBlk
            t1 = T;
        else
            t1 = t0+cut-1;
        end
        dat = datIn(:,:,t0:t1);

        datMA = movmean(dat,movAvgWin,3);
        datMin = min(datMA,[],3)-xBias;
        dF(:,:,t0:t1) = dat - datMin;
    end

end







