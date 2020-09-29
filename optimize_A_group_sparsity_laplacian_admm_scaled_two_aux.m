function [A_hat,Y_hat,Z_hat] =  optimize_A_group_sparsity_laplacian_admm_scaled_two_aux(F_p,A,X,Y,Z,lambda0,L,maxit,printlevel,tol)
%% min_{A,Z}  ||F- A^T.X||^2  + lambda0/2*Trace(A*L_a*A') s.t. AA^T = I

% Introduce auxilliary variables.
a = A(:); c = a; d = a;
dim = size(A,2);

% Step size
muu = 10^-3;

% Initialize lagrange mutipliers.
y = Y(:);
z = Z(:);

iter = 0;


if (printlevel == 1)
    fprintf('\n \t iteration \t Obj \t constraints \t aux1_feas \t aux2_feas \n \t ...---------------------------------------------- \n');
end
j=1;
iter = 1;
res = 10^6;
yy=[];
while((iter< maxit)&& res > 10^-5)
    iter = iter+1;
    %% Optimize A 
    % Lagrange multipiers.
    Z = reshape(z,[length(a)/dim,dim]);
    Y = reshape(y,[length(a)/dim,dim]);
   
    % Auxilliary variables.
    C = reshape(c,[length(a)/dim,dim]);
    D = reshape(d,[length(a)/dim,dim]);
    
    [U,S,V] = svd(C+D+Y+Z);
    A = U*eye(size(S))*V';
        
    %% replace a with update
    a = A(:);   

    
    % Update for C.
    C = (X*X' + (1/muu)*eye(size(A,1)))\(X*F_p'+ (1/muu)*(A-Z));
    c = C(:);
    
    % Update for D.
    D = ((1/muu)*(A-Y))/(lambda0*L' + (1/muu)*eye(dim));
    d = D(:);
     
     
    %Constraint tolerance.
    eta = muu^(0.1 + 0.9*j); 
    p = [c-a;d-a];

      if eta <=10^-3
         eta = 10^-3;
      end
    
      if(norm(p) <= eta)
            Y = Y + (D-A);
            y = Y(:);
            
            Z = Z + (C - A);
            z = Z(:);
      else
         Y = Y + (D-A);
         y = Y(:);
            
         Z = Z + (C - A);
         z = Z(:);
         muu = max(min(0.9*muu,muu^1.1),10^-3);
         j=j+1;
      end
    
        %iter = iter+1;
        obj = fun_primal(a,F_p,X,L,lambda0,dim);
         
        res1 = [c-a;d-a];
        
        res3 = -X*(F_p' - X'*C) + (1/muu)*Z;
        res3 = res3(:);
        
        res4 = lambda0*(D*L') + (1/muu)*Y;
        res4 = res4(:);
        
        res = norm([res1;res3;res4],2);
        
         if (printlevel == 1)
            fprintf('\t %03d \t \t %0.5f \t %0.9f \t %0.9f \t  %d \n',iter,obj,norm(res1),norm(res3),norm(res4));
            yy(iter-1) = obj;
            figure(100), plot(1:iter,yy);
         end

    
    
end
A_hat = reshape(a,[length(a)/dim,dim]);

Y_hat = reshape(y,[length(a)/dim,dim]);
Z_hat = reshape(z,[length(a)/dim,dim]);

end


% Objective Function
function obj= fun_primal(z,F_p,X,L,lambda0,dim)

Z = reshape(z,[length(z)/dim,dim]);

% objective calc.
obj = norm(F_p - Z'*X,'fro')^2 + (lambda0/2)*trace(Z*L*Z');

end
