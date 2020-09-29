function b_hat = optimize_b(X,class,b,lambda0,lambda5)

%%  min_{b} - Entropy(.) + lambda5/2*||b||^2    

options = optimoptions('fmincon','Algorithm', ...
          'trust-region-reflective','SpecifyObjectiveGradient',true,'Display','off');%,'HessianFcn','objective');

% options = optimoptions('fmincon','SpecifyObjectiveGradient',true);

Aa=[]; bb=[]; Aeq=[]; beq=[];  nonlcon = [];

x = b(:);
lb= [];

ub=[];

x=fmincon(@(x)cost_b(x,X,class,b,lambda0,lambda5),x,Aa,bb,Aeq,beq,lb,ub,nonlcon,options);
b_hat = x;


end
function [out,g] = cost_b(x,X,class,b,lambda0,lambda5)
    b = reshape(x,size(b));
    
    o_1 = -lambda0*(sum( (class.*log(sigmoid((X)'*b)+eps)) + ((1-class).*log( (1-sigmoid((X)'*b)) + eps ))  ));
    
    o_2 = (lambda5/2)*(norm(b,2)^2);
    
    out = 0.5*(o_1 + o_2) ;
    
    g = (lambda0/2)*(X*(sigmoid((X)'*b) - class)) + (lambda5/2)*b;
    g=g(:);

end
