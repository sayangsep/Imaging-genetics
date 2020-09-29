function final_cost = cost_fn_genetic_classification(F,A,C,X,G,class,b,L_a,lambda0,lambda1,lambda2,lambda3,lambda4)

%%  min_{A,X,b,B} ||F- A^T.X||^2  - Entropy(.)+ ||G - C^T.X||^2 ... 
 %              + lambda1*Tr(A*L_a*A')  + lambda5*(C*L_g*C')  +   lambda2*||C||_1 +
 %                lambda3*|| X ||_F
 %                lambda4*||b||^2

o_1  = (norm((F - A'*X),'fro'))^2;

o_2  = -lambda0*sum( (class.*log(sigmoid(((X))'*b + eps))) + ((1-class).*log( (1-sigmoid(((X))'*b)) + eps ))  );

o_4  = norm(G-C'*X,'fro')^2;

o_5  = (lambda1/2)*trace(A*L_a*A');

o_7  = (lambda2)*sum(sqrt(sum(C'.^2,2)));

o_8  = (lambda3/2)*norm((X),'fro')^2;

o_10 = (lambda4/2)*(norm(b,2)^2);

final_cost = 0.5*(o_1 + o_2 + o_4 + o_5 + o_7 + o_8  + o_10);
end
