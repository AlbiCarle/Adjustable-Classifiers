clc; clear all; close all;

%% PLATOONING

I = readtable('./Dataset/platooning.txt');

X = table2array(I);

rand_shuffle = randperm(size(I, 1));
X = X(rand_shuffle, :);
Y = X(:,size(X,2));
X = X(:,1:size(X,2)-1);

n_feats=size(X,2);

X_Y = [X,Y];

X1 = X_Y(Y==0,:);
X2 = X_Y(Y==1,:);


X_learn = [X1(1:size(X1,1),1:n_feats);
           X2(1:size(X2,1),1:n_feats)];
Y_learn = [1*ones(size(X1,1),1);
           -ones(size(X2,1),1)];
%%%%%%%%%%%% XXXXX %%%%%%%%%%%%

[Z, C, S] = normalize(X_learn,'norm',Inf); % for de-normalizing: X = Z.*S + C

%%%%%%%%%%%% XXXXX %%%%%%%%%%%%

delta = 1E-6; 
Prob = [];
J = [];
metrics = [];
Epsilon = [0.01];

for epsilon = Epsilon

n_c = (7.47)/epsilon*log(1/delta);
r = ceil(epsilon*n_c*0.5);
n_tr = 200;
n_ts = 5000;

[Xtr, Ytr, Xts, Yts, Xcl, Ycl] = split_dataset(Z, Y_learn, n_tr, n_ts, n_c);

Xcl_U = Xcl(Ycl == -1,:); Ycl_U = Ycl(Ycl == -1);
Xcl_S = Xcl(Ycl == +1,:); Ycl_S = Ycl(Ycl == +1);

n_U = size(Xcl,1);

%%%%%%%%%%%% XXXXX %%%%%%%%%%%%

kernel = 'gaussian';
param = 0.15; % 0.11

Eta = [0.01, 0.1, 1];
Tau = [0.1, 0.5, 0.9];

for eta = Eta
    for tau = Tau

%%%%%%%%%%%% XXXXX %%%%%%%%%%%%

alpha = LogRegr_train(Xtr, Ytr, kernel, param, tau, eta);
    
R = offset(Xtr, Ytr, alpha, kernel, param, eta, tau);

y_pred = LogRegr_test(Xtr, Ytr, Xtr, alpha, R, 0, kernel, param, eta);

[TPR, FPR, TNR, FNR, F1, ACC] = ConfusionMatrix(Ytr, y_pred, 'on');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R = offset(Xtr, Ytr, alpha, kernel, param, eta, tau);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gamma_rho = barrhoSVM(Xtr, Ytr, Xcl_U, R, alpha, kernel, param, eta);

Gamma_rho_sort = sort(Gamma_rho,'descend');

rho_star = Gamma_rho_sort(r);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_star_pred = LogRegr_test(Xtr, Ytr, Xtr, alpha, R, rho_star, kernel, param, eta);
y_star_pred_cl = LogRegr_test(Xtr, Ytr, Xcl_S, alpha, R, rho_star, kernel, param, eta);

x_U = Xtr(y_star_pred == -1,:);
x_S = Xtr(y_star_pred == +1,:);

PSR = sum(y_star_pred(y_star_pred == +1) );
PSR_cl = sum(y_star_pred_cl(y_star_pred_cl == +1));

m = [Ytr y_star_pred];

FN = sum((m(:,1)==-1 & m(:,2)==+1));
TP = sum((m(:,1)==+1 & m(:,2)==+1));

prob = FN/PSR
Prob = [Prob;prob];

s_points = TP/PSR;

n_points = [J;PSR_cl]

metrics = [metrics,[prob,n_points]];

    end
end
end




