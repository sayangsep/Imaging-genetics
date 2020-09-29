function [beta1,beta2,F_out] = generate_training_sdmt(F,id_train)

info = readtable('data/sdmt/subject_demographic.csv');
X = [info.age,info.edu_completed,info.wratreadingstandardscore];

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

for i=1:length(id_train)
F_out(:,i) = F_reg(2,:,i)'  - F_reg(1,:,i)';
end

end

