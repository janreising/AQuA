clc;

% use this to see memory as well
% profile('-memory','on');
              
%% create a dummy matrix
% test case
%A = randi(100,100, 100, 100); % 8 MB
% realistic case
% A = randi(100, 50, 50, 100000); % 11.52 GB
A = randi(100, 1200, 1200, 200); % 11.52 GB
A((10 < A) & (A < 30)) = NaN;

w = whos("A");
fprintf(strcat("Matrix size: ",num2str(w.bytes/1024/1024/1024),"GB\n"));

%% run function
tic;
B = imputeMov(A);
toc;

tic;
C = wiktMov(A);
toc;
compare("Wiktor", B, C);
C=NaN;

tic;
C = antMov(A);
toc;
compare("Antoine", B, C);
C=NaN;

tic;
C = combined(A);
toc;
compare("Combined", B, C);
C=NaN;

% original function
function datx=imputeMov(datx)

T = size(datx,3);

% > forward sweep replaces NaN values 
% > with previous value
for tt=2:T
    tmp = datx(:,:,tt);
    tmpPre = datx(:,:,tt-1);
    idx0 = find(isnan(tmp(:)));
    tmp(idx0) = tmpPre(idx0);
    datx(:,:,tt) = tmp;
end

% > reverse sweep replaces NaN values 
% > with next value
for tt=T-1:-1:1
    tmp = datx(:,:,tt);
    tmpNxt = datx(:,:,tt+1);
    idx0 = find(isnan(tmp(:)));
    tmp(idx0) = tmpNxt(idx0);
    datx(:,:,tt) = tmp;
end

for tt=1:size(datx,3)
    datx(:,:,tt) = imgaussfilt(datx(:,:,tt),[1 1]);
end
end


function datx=wiktMov(datx)
    [H,W,Z] = size(datx);
    % > forward sweep replaces NaN values 
    % > with previous value
    datx = reshape(datx, H*W, Z);
    nans = isnan(datx);
    for tt=2:Z
        datx(nans(:,tt),tt) = datx(nans(:,tt),tt-1);
    end
    % > reverse sweep replaces NaN values 
    % > with next value
    nans = isnan(datx);
    for tt=Z-1:-1:1
        datx(nans(:,tt),tt) = datx(nans(:,tt),tt+1);
    end
    datx = reshape(datx,H,W,Z);
    for tt=1:Z
        datx(:,:,tt) = imgaussfilt(datx(:,:,tt),[1 1]);
    end
end

%% ANTOINE

function A=ffill(A)
    % > forward sweep replaces NaN values 
    % > with previous value

    for t=2:size(A,3)
        At1=A(:,:,t-1);
        At=A(:,:,t);
        idx = isnan(At);
        At(idx) = At1(idx);
    end
end
function A=bfill(A)
    % > forward sweep replaces NaN values 
    % > with previous value
    for t=size(A,3)-1:-1:1
        At1=A(:,:,t+1);
        At=A(:,:,t);
        idx = isnan(At);
        At(idx) = At1(idx);
    end
end

function A=gaussFilt(A)
    %h = fspecial('gaussian', [5,5], [1,1]);
    parfor tt=1:size(A,3)
        A(:,:,tt) = imgaussfilt(A(:,:,tt),[1 1], 'FilterDomain','spatial');
        %A(:,:,tt) = imfilter(A(:,:,tt),h,'conv');
    end
end

% original function
function A=antMov(A)
    A=ffill(A);
    A=bfill(A);
    A=gaussFilt(A);
end

%% Combined
function datx=combined(datx)

    [H,W,Z] = size(datx);
    % > forward sweep replaces NaN values 
    % > with previous value
    datx = reshape(datx, H*W, Z);
    nans = isnan(datx);
    for tt=2:Z
        datx(nans(:,tt),tt) = datx(nans(:,tt),tt-1);
    end
    % > reverse sweep replaces NaN values 
    % > with next value
    nans = isnan(datx);
    for tt=Z-1:-1:1
        datx(nans(:,tt),tt) = datx(nans(:,tt),tt+1);
    end
    datx = reshape(datx,H,W,Z);
    parfor tt=1:Z
        datx(:,:,tt) = imgaussfilt(datx(:,:,tt),[1 1]);
    end

end

function compare(name, A,B)

    if isequal(A,B) == 1
        fprintf(strcat(name,": Results are equal\n"));
    else
        fprintf(strcat(name,": UNEQUAL (!)\n"));
    end

end
