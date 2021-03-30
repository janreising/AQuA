clc;

A = randi(100,1200, 1200, 100);
A(2,2,2) = NaN;
A(2,2,3) = NaN;
A(2,2,6) = NaN;

CA = codistributed(A,codistributor1d(3));

% disp(A);

tic;
M = my_func(A);
toc;

% disp(A);


B = A(:,:,:);

% fprintf("A");
% disp(A);
% fprintf("B");
% disp(B);

% C = A.*B;
% disp(C);

% D = fillmissing(A, 'previous', 3);
% D = fillmissing(D, 'next', 3);
% fprintf("D");
% disp(D);

% tic;
% B = img.imputeMov(B);
% toc;
% fprintf("B");
% disp(B);

if isequal(B,M) == 1
    fprintf("Is EQUAL\n");
else
    fprintf("UNEQUAL!\n");
end

function B = my_func(B)

[W, H, T] = size(B);

for x=1:W
    
    slice = B(x,:,:);
    for y=1:H
   
        ind = find(isnan(slice(1, y,:)));
        if isempty(ind) == 0
            
            n=1;
            while n<length(ind)+1
               
                val0 = ind(n);
                val = ind(n);
                m=1;
                while m<length(ind)-n
                    val_next = ind(n+m);
                    
                    if val_next == val+1
                        val = val_next;
                    else
%                         n = m-1; %TODO no idea how that is to be updated
                        break 
                    end
                    
                    m = m+1;
                end
                
                if val0 == 1
                    B(x,y,val0:val) = B(x, y, val+1);
                else
                    B(x,y,val0:val) = B(x, y, val0-1);
                end

%                 slice(1, y,val0:val) = slice(1, y, val0-1);
                
                n=n+m;
                
            end
        end
    end
    
    B(x,:,:) = slice;
end

end