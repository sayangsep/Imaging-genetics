function A = optimize_C_l21_fixed_point(F_p,A,X,lambda0)

A = A';
y=[];
V = size(F_p,1);
dim = size(A,2);

%% Initialize variables.
A_temp = A;
epss = 10^-16;
d0 =((sum(A.^2,2)));
d0(find(d0<epss)) = d0(find(d0<epss)) + epss;
d0 = 2*(d0.^0.5);

d_temp = 1./d0;
t=1;
res = 10^6;

%% Fixed point iteration.

while(t<500 && res > 10^-6)
   
 [y(t),g] = cost_A(F_p,A,X,lambda0,epss);
 res = norm(g,'fro');

%% Update all the basis vectors.

for i=1:size(A,1)
    A_temp(i,:) = (F_p(i,:)*X')/(X*X' + (lambda0*(d_temp(i)))*eye(dim)) ;
end

dd = ((sum(A_temp.^2,2)));
dd(find(dd<epss)) = dd(find(dd<epss)) + epss;
dd = 2*(dd.^0.5);
d_temp = 1./dd;

A = A_temp; 
if (t>100)
if(norm((A_temp-A),'fro')/max(norm(A,'fro'),1)<10^-8)
    %disp(sprintf('Iteration=%d',t));
    break;
end
end



t = t+1;

epss = 0.1*epss;
end
A = A';
end
function [out,grad] = cost_A(F_p,A,X,lambda0,epss)
    o_1 = (lambda0/2)*sum(sum(A.^2,2).^(0.5));
    o_2 = 0.5*(norm((F_p-A*X),'fro'))^2;
    out = o_1 + o_2;
    
    l2 = sqrt(sum(A.^2,2));
    l2((l2<10^-32)) = 10^-32;
    grad = -(F_p - A*X)*X' + (lambda0/2)*(diag(1./l2)*A);
    
end
