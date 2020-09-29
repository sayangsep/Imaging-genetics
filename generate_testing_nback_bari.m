function [F_out] = generate_testing_nback_bari(F,id_perm,id_test,b1,b2)
info = readtable('/home/sayan/myshare/Users/sghosal3/data/bari/nc_scz_nback_bari_matched_final.csv');
X = [info.age,info.TIB]; 
%%  To permute the data.
X = X(id_perm,:);

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
