function X_hat = optimize_X_imaging_genetics(F_p,A,X,G,C,class,b,lambda3)
  
V = size(F_p,1);
n_p = size(F_p,2);
dim = size(A,1);


options = optimoptions('fmincon','Algorithm', ...
          'trust-region-reflective','SpecifyObjectiveGradient',true,'Display','off');%,'HessianFcn','objective');

% options = optimoptions('fmincon','SpecifyObjectiveGradient',true);

Aa=[]; bb=[]; Aeq=[]; beq=[];  nonlcon = [];
for t = 1:size(X,2)
x = X(:,t);
lb= [];

ub=[];

x=fmincon(@(x)cost_X(x,F_p(:,t),A,G(:,t),C,X(:,t),class(t),b,lambda3),x,Aa,bb,Aeq,beq,lb,ub,nonlcon,options);
X_hat(:,t) = x;
end

end
%%  min_{X} ||F- A^T.X||^2  - Entropy(.) + ||G - C^T.X||^2 + ...
    %                 lambda3/2*|| X ||_F
    
function [out,g] = cost_X(x,F,A,G,C,X,class,b,lambda3)
    X = reshape(x,size(X));
    
    %First Term
    o_1 = (norm((F - A'*X),'fro'))^2;
    
    %Second Term
    o_2 = norm(G -C'*X,'fro')^2;
    
    %Third Term
    o_3 = -2*(sum( (class.*log(sigmoid((X)'*b) + eps)) + ((1-class).*log( (1-sigmoid((X)'*b)) + eps ))  ));
    
    %Fourth Term
    o_4 = (lambda3/2)*norm((X),'fro')^2;
    
    out = o_1 + o_2 + o_3 + o_4;
    
    g = -2*A*(F - A'*X) - 2*C*(G - C'*X) + 2*b*(sigmoid((X)'*b) - class)' + (lambda3)*(X);
    g=g(:);

end
