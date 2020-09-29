function [beta1,beta2,F_out] = generate_training_nback_bari(F,id_perm,id_train)
info = readtable('/home/sayan/myshare/Users/sghosal3/data/bari/nc_scz_nback_bari_matched_final.csv');
X = [info.age,info.TIB]; 
%%  To permute the data.
X = X(id_perm,:);

% id_train = 1:106;

%% To arrange data according to training.
X = X(id_train,:);

F = F(:,:,id_train);

for id=1:246
   
Y = permute(F(1:2,id,:),[3,1,2]);
b1 = glmfit(X,Y(:,1),'normal','constant','off');
b2 = glmfit(X,Y(:,2),'normal','constant','off');

% Y = permute(F_scz_b(2,id,:)-F_scz_b(1,id,:),[3,1,2]);
% b = glmfit(X,Y(:,1),'normal','constant','off');
% F_scz_b_reg(id,:) = permute((Y-X*b),[2,1]);

F_reg(:,id,:) = permute((Y-[X]*[b1 b2]),[2,3,1]);

beta1(:,id) = b1;
beta2(:,id) = b2;

end

%% Debugging
% F_reg = F_reg(:,:,id_trainn);

for i=1:length(id_train)
F_out(:,i) = F_reg(2,:,i)'  - F_reg(1,:,i)';
end

end

