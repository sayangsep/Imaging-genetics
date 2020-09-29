function [] = cross_validation_script_libd_nback(dd,ll0,ll1,ll2,ll3,ll4,idd)
%% Comment or uncomment the following 3 lines to exactly replicate our results.
rng('default');
load('random_state_nback.mat','state');
rng(state)
%%%%%%%%%%%%%%%%%%5

addpath(genpath(pwd));
cd(pwd);

% number of ROIs
V = 246;
% number of subjects
nn = 106;


%% load brain activation ROIs
n2_back = readtable('data/nback/N2_Back_Activation.csv');
n2_back = table2array(n2_back(:,2:end));
n0_back = readtable('data/nback/N0_Back_Activation.csv');
n0_back = table2array(n0_back(:,2:end));

% concatenation of N2-Back and N0-Back to obtaint the contrast maps
F(1,1:V,1:nn) = n0_back';
F(2,1:V,1:nn) = n2_back';


%% Load SNP data.
snp = readtable('data/nback/N_Back_SNP_data.csv');
snp = table2array(snp(:,2:end));
G = snp';

%% Load Diagnosis
info = readtable('data/nback/subject_demographic.csv');
class = info.diagnosis;

%% Initialize the output variables and class variable.
ppp = {};
ss = {};

%%  min_{A,X,b,B} ||F- A^T.X||^2  - Entropy(.)+ ||G - C^T.X||^2 ... 
 %              + lambda1/2*Tr(A*L_a*A')  + lambda2*||C||_21 +
 %                lambda3/2*|| X ||_F
 %                lambda4/2*||b||^2
 
% F = V x nn, A = dim_X x 246, X_p = dim_X x nn, b = dim_X x 1, G = g x nn, C = dim_X x g  

%% Parameters sweep.
dime = [1:50]; % no.  of latent dimensions.

lambda0_c = [2];

lambda1_c =[2.5,5,10,12.5,15,17.5,20,40,80]; % graph regularizer

lambda3_c = 2*[0.15,0.3,0.6,1.2,2.4,4.8]; %l2 regularizer over latent variable. 

lambda2_c = [2.5,5,10,15,20,25,40,80]; % group sparsity regularizer over gene bases.

lambda4_c = 2*[0.015,0.03,0.06,0.12,0.24,0.48]; %l2 regularizer over regression vector. 


fold = 10; % no. of folds.

class_cv_hat       =    zeros([nn,length(dime),length(lambda0_c),length(lambda1_c),...
                        length(lambda2_c),length(lambda3_c),length(lambda4_c)]);
                    
entropy_final      =    zeros([length(dime),length(lambda0_c),length(lambda1_c),length(lambda2_c),...
                        length(lambda3_c),length(lambda4_c)]);
                    
overlap_final      =    zeros([length(dime),length(lambda0_c),length(lambda1_c),length(lambda2_c),...
                        length(lambda3_c),length(lambda4_c)]);


%% Dimension Sweep
dim = dd;

lambda0 = lambda0_c(ll0);

lambda1 = lambda1_c(ll1);

lambda2 = lambda2_c(ll2);

lambda3 = lambda3_c(ll3) ;       
        
lambda4 = lambda4_c(ll4);

param.dim = dd; 
param.lambda0 = lambda0; 
param.lambda1 = lambda1;
param.lambda2 = lambda2;
param.lambda3 = lambda3;
param.lambda4 = lambda4;

%% starting of cross validation.
for c = 1:fold+1 % parallelize the CV folds.

id_cv = (floor(nn/fold)*(c-1)+1:floor(nn/fold)*(c-1)+floor(nn/fold));
if(c==fold+1)
    id_cv = (floor(nn/fold)*(c-1)+1:nn);
end

id_train = find(ismember([1:nn], id_cv)==0);

%% Generate Training data. Here we are regressing out the demographics.
[beta1,beta2,F_train] = generate_training_nback(F,id_train);

G_train = G(:,id_train);

%% Normalize gnetics component.
mean_G_train = mean(G_train,2);
G_train = G_train - mean_G_train;

class_train  = (class(id_train));

%% Generate sample correlation matrix for brain activations.
[C,~] = corr(F_train');
C = C.*(C>0);
L_a = diag(sum(C,2)) - C;


%% Initialize variables.

as = randn(size(F_train,1),dim);
[A,~] = qr(as);
A = A(:,1:dim);
A = A';
Y = rand(dim,size(F_train,1));
Z = rand(dim,size(F_train,1));
C = rand(dim,size(G_train,1));
X = rand(dim,length(id_train)); 
b = rand(dim,1);
iter = 1;
y=[];

while(iter<100)
    
%%  min_{A,X,b,B}   ||F- A^T.X||^2  ... 
%                + lambda1/2*Tr(A*L_a*A') s.t. AA^T = I
 
[A_temp,Y,Z]  = optimize_A_group_sparsity_laplacian_admm_scaled_two_aux(F_train,A,X,Y,Z,lambda1,L_a,3000,0,1e-4);

cost_fn_genetic_classification(F_train,A_temp,C,X,G_train,class_train,b,L_a,lambda0,lambda1,lambda2,lambda3,lambda4)


       
%%  min_{C}           ||G - C^T.X||^2 ... 
%                  +  lambda2*||C||_l21   
C_temp = optimize_C_l21_fixed_point(G_train,C,X,lambda2);

cost_fn_genetic_classification(F_train,A_temp,C_temp,X,G_train,class_train,b,L_a,lambda0,lambda1,lambda2,lambda3,lambda4)


%%  min_{X}       ||F- A^T.X||^2  - Entropy(.) + ||G - C^T.X||^2 + ... 
%                 lambda3/2|| X ||_F
X_temp = optimize_X_imaging_genetics(F_train,A_temp,X,G_train,C_temp,class_train,b,lambda0,lambda3); 

cost_fn_genetic_classification(F_train,A_temp,C_temp,X_temp,G_train,class_train,b,L_a,lambda0,lambda1,lambda2,lambda3,lambda4) 
             

%%  min_{A,X,b,B} -Entropy(.) ...
 %                lambda4/2*||b||^2  
b_temp = optimize_b(X_temp,class_train,b,lambda0,lambda4);

cost_fn_genetic_classification(F_train,A_temp,C_temp,X_temp,G_train,class_train,b_temp,L_a,lambda0,lambda1,lambda2,lambda3,lambda4)

   
y(iter) = cost_fn_genetic_classification(F_train,A_temp,C_temp,X_temp,G_train,class_train,b_temp,L_a,lambda0,lambda1,lambda2,lambda3,lambda4);

    
    
    
   %% UPDATE  
    A = A_temp;
    b = b_temp;
    X = X_temp;
    C = C_temp;

     
    iter = iter+1;
    
    % Terminate loop when the change is not too big.
    if(iter>5)
    if(abs((y(end) - y(end-1)))/abs(y(1)-y(2))<10^-3)
        break;
    end
    end



%     figure(2),subplot(1,2,1), imagesc(F_train);title('Original');colormap('jet');caxis([min(F(:)),max(F(:))]);colorbar;
%     figure(2),subplot(1,2,2), imagesc(A'*X);title('Reconstructed');colormap('jet');colorbar;%caxis([min(F(:)),max(F(:))]);colorbar;
 


end

update(c) = iter;
%% Cross Validation
F_cv = generate_testing_nback(F,id_cv,beta1,beta2);
G_cv = G(:,id_cv);
G_cv = G_cv - mean_G_train;
class_cv  = (class(id_cv));
X_cv =  rand(dim,length(id_cv));


%%  min_{X} 0.5*||F- A^T.X||^2  + ||G - C^T.X||^2 + ... 
%                 lambda3/2*|| X ||_F
X_cv = optimize_X_cv(F_cv,A,C,G_cv,lambda3);
    
% figure(12),subplot(2,3,1),imagesc(X),colormap('jet'),caxis([min(X(:)) max(X(:))]),colorbar,title('train');
% figure(12),subplot(2,3,2),imagesc(X_cv),colormap('jet'),caxis([min(X(:)) max(X(:))]),colorbar,title('X CV');
% figure(12),subplot(2,3,3),imagesc(X_cv_tr),colormap('jet'),caxis([min(X(:)) max(X(:))]),colorbar,title('X CV tr');
% figure(12),subplot(2,3,4),imagesc(b),colormap('jet'),colorbar,title('b');
% figure(12),subplot(2,3,5),imagesc(A'),colormap('jet'),colorbar,title('A^T');
% figure(12),subplot(2,3,6),imagesc(C'),colormap('jet'),colorbar,title('C^T');
% figure(13),plot(1:1246,sum(C'.^2,2),'-o');

%% Save fold resutls
ppp{c} = class_cv';
ss{c} =  sigmoid(X_cv'*b)';



% zz{c} = sigmoid(X_cv_tr'*b)';

% figure(6), plot([1:id_cv(end)],cell2mat(ss),'o',[1:id_cv(end)],cell2mat(zz),'*',[1:id_cv(end)],cell2mat(ppp),'+');
%   xlabel('Indices');ylabel('predicted class');
% title(sprintf('CV dim = %s, l_3=%s, l_1=%s,l_2=%s',num2str(dim),num2str(lambda3),num2str(lambda1),num2str(lambda2)));
% % 


end


%% Classification performance results.
ss = cell2mat(ss);
perf_str = classperf(class,ss>0.5);
perf = perf_str.CorrectRate;

class_cv_hat(:,dd,ll0,ll1,ll2,ll3,ll4) = ss(:)';

entropy_final(dd,ll0,ll1,ll2,ll3,ll4) =  -(sum( (class(:).*log(ss(:))) + ((1-class(:)).*log( (1-ss(:)) ))  ));

overlap_final(dd,ll0,ll1,ll2,ll3,ll4) = perf;
 %% Save the results in a file indexed by idd.
save(sprintf('classification_results_nback/classification_imaging_genetics_l21_%d.mat',idd),'class_cv_hat','entropy_final','overlap_final','param');

end
