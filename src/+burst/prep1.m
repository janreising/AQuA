function [dat,opts] = prep1(p0,f0,rgT,opts,ff,indices)
    %PREP1 load data and estimation noise
    % TODO: use segment by segment processing to reduce the impact of bleaching
    
    bdCrop = opts.regMaskGap;
    
    [filepath,name,ext] = fileparts([p0,filesep,f0]);
    opts.filePath = filepath;
    opts.fileName = name;
    opts.fileType = ext;
    
    % read data
    fprintf('Reading data\n');
    if strcmp(ext,'.mat')
        file = load([p0,filesep,f0]);
        headers = fieldnames(file);
        dat = file.(headers{1});
        maxImg = -1;
    elseif endsWith(f0, '.h5')
        fprintf('Loading h5 file\n');
        
        if ~exist('indices','var')
            indices = [1 Inf];
        end
        
        disp(h5info(strcat(p0,f0)));
        dat = h5read(strcat(p0,f0),'/dff/ast', [1 1 indices(1)], [Inf Inf indices(2)]);
        maxImg = -1;
        fprintf('Loading finished.\n');
    else
        [dat,maxImg] = io.readTiffSeq([p0,filesep,f0]);
    end
    
    if exist('rgT','var') && ~isempty(rgT)
        dat = dat(:,:,rgT);
    end
    dat = single(dat);
    maxDat = max(dat(:));
    dat = dat/maxDat;
    dat = dat(bdCrop+1:end-bdCrop,bdCrop+1:end-bdCrop,:);
    dat(dat<0) = 0;
    if opts.usePG==1
        dat = sqrt(dat);
    end
%    if exist('ff','var')
%        waitbar(0.4,ff);
%    end
    
%     dat = dat + randn(size(dat))*1e-4;
    [H,W,T] = size(dat);
    opts.sz = [H,W,T];
    opts.maxValueDepth = maxImg;
    opts.maxValueDat = maxDat;
    
end




