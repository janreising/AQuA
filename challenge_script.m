clc;

% use this to see memory as well
% profile('-memory','on');
              
%% create a dummy matrix
% test case
A = randi(100,100, 100, 100); % 8 MB
% realistic case
% A = randi(100, 1200, 1200, 1000); % 11.52 GB

%% run function
tic;
% its a bit hard to keep track but it uses
% approximately 300% of the matrix size in
% additional runtime memory
A = imputeMov(A);
toc;

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