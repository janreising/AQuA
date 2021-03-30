clc;
clear all;
% use this to see memory as well
% profile('-memory','on');
              
%% create a dummy matrix
% test case
T=100;
w=1000;
h=w;
mvalue=100;

Xin = rand(w, h, T); % 8 MB
% realistic case
% A = randi(100, 1200, 1200, 1000); % 11.52 GB
p = .1;
Xin(Xin<p) = nan;
Xin = round(mvalue*Xin);

%% run function
tic;
% its a bit hard to keep track but it uses
% approximately 300% of the matrix size in
% additional runtime memory
Xin = imputeMov(Xin);
toc;
%profile viewer

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
function A=imputeMov(A)
    A=ffill(A);
    A=bfill(A);
    A=gaussFilt(A);
end


