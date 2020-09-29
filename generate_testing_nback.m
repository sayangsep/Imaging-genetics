function [F_out] = generate_testing_nback(F,id_test,b1,b2)

info = readtable('data/nback/subject_demographic.csv');
X = [info.age,info.edu_completed,info.wratreadingstandardscore,info.percor_2back]; 

%% To arrange data according to testing.
X = X(id_test,:);

F = F(:,:,id_test);

for id=1:246
   Y = permute(F(1:2,id,:),[3,1,2]);
F_reg(:,id,:) = permute((Y-[X]*[b1(:,id) b2(:,id)]),[2,3,1]);

end

for i=1:length(id_test)
F_out(:,i) = F_reg(2,:,i)'  - F_reg(1,:,i)';
end

end
