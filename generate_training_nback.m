function [beta1,beta2,F_out] = generate_training_nback(F,id_train)

info=readtable('data/nback/subject_demographic.csv');
X = [info.age,info.edu_completed,info.wratreadingstandardscore,info.percor_2back]; 

%% To arrange data according to training.
X = X(id_train,:);

F = F(:,:,id_train);

for id=1:246
   
Y = permute(F(1:2,id,:),[3,1,2]);
b1 = glmfit(X,Y(:,1),'normal','constant','off');
b2 = glmfit(X,Y(:,2),'normal','constant','off');

F_reg(:,id,:) = permute((Y-[X]*[b1 b2]),[2,3,1]);

beta1(:,id) = b1; % regression coefficients
beta2(:,id) = b2; % regression coefficients

end

for i=1:length(id_train)
F_out(:,i) = F_reg(2,:,i)'  - F_reg(1,:,i)'; % contrast maps.
end

end

