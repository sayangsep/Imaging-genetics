function X_hat = optimize_X_cv(F,A,C,G,lambda3)

    %%  min_{X} ||F- A^T.X||^2  + ||G - C^T.X||^2 + ... 
    %                 lambda3/2*|| X ||_F
    
dim = size(A,1);

t1 = ((A*A') + (C*C') + (lambda3/2)*eye(dim));
t2 = ( A*F + C*G);
% X_hat = pinv(t1)*t2;
X_hat = t1\t2;
end