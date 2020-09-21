function X_hat = optimize_X_cv(F,A,C,G,lambda3)

    %%  min_{X} ||F- A^T.X||^2  + ||G - C^T.X||^2 + ... 
    %                 lambda3/2*|| X ||_F
    
dim = size(A,1);

t1 = (2*(A*A') + 2*(C*C') + lambda3*eye(dim));
t2 = ( 2*A*F + 2*C*G);
X_hat = t1\t2;
end