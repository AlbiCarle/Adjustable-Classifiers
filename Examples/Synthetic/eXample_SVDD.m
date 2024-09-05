addpath './Conformal_Stuff'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;

% Data

mu1 = [+1;+1]; sigma1 = 1;
mu2 = [-1;-1]; sigma2 = 1;

N = 1000;

p_A = 0.5; p_O = 0.1;

[Xtr,Ytr] = generate_data(N, p_A, p_O, mu1, mu2, sigma1, sigma2);

figure(1)

gscatter(Xtr(:,1), Xtr(:,2), Ytr, 'br')

kernel = 'gaussian';
param = 1.5;

tau = 0.5; 
eta = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = SSVDD_Train(Xtr, Ytr, kernel, param, tau, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R = Radius(Xtr, Ytr, alpha, kernel, param, eta, tau);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xts,Yts] = generate_data(5000, p_A, p_O, mu1, mu2, sigma1, sigma2);

yts_predict = SSVDD_Test_conformal(Xtr, Ytr, Xts, alpha, R, kernel, param, eta);

mycolor = [0,1,0];

figure(1)

gscatter(Xts(:,1), Xts(:,2), Yts, 'br')

hold on

plot_SSVDD_conformal(Xtr, Ytr, Xts, alpha, R, kernel, param, eta, mycolor);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xcl,Ycl] = generate_data(5000, p_A, p_O, mu1, mu2, sigma1, sigma2);

ycl_predict = SSVDD_Test_conformal(Xtr, Ytr, Xcl, alpha, R, kernel, param, eta);

n = size(Xcl,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

scores = scoreSVDD(Xtr, Ytr, Xcl, Ycl, R, alpha, kernel, param, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

barrhos = barrho(Xtr, Ytr, Xcl, R, alpha, kernel, param, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all

epsilon = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

qhat = quantile(scores, ceil((n+1)*(1-epsilon))/n); %%% !!!!!!!!!!!!! 1-epsilon

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

score_ts_plus = scoreSVDD(Xtr, Ytr, Xts, +1, R, alpha, kernel, param, eta);

score_ts_minus = scoreSVDD(Xtr, Ytr, Xts, -1, R, alpha, kernel, param, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Cset = [score_ts_plus-qhat <= 0,(score_ts_minus-qhat <= 0)*2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Cscatter = Cset(:,1) + Cset(:,2);

unique(Cscatter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


scatterColors = {'m', 'g','r','y'};
scatterLegend = {'$\{\emptyset\}$', '$\{+1\}$', '$\{-1\}$', '\{-1,+1\}'};
myScatterColor = scatterColors(unique(Cscatter)+1); 
myScatterLegend  = scatterLegend(unique(Cscatter)+1); 

figure(2)

gscatter(Xts(:, 1), Xts(:, 2),Cscatter, cell2mat(myScatterColor), '.');

legend(myScatterLegend,'Interpreter','latex','FontSize',10);

axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S_epsilon = [score_ts_plus-qhat <= 0,(score_ts_minus-qhat > 0)];
S_epsilonscatter = S_epsilon(:,1).*S_epsilon(:,2);

figure(3)

gscatter(Xts(:, 1), Xts(:, 2),S_epsilonscatter, 'rg', '.');

legend({'','$\mathcal{S}_\varepsilon$'},'Interpreter','latex','FontSize',10);

axis equal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qhat
rho_epsilon = abs(qhat);

figure(4)

gscatter(Xts(:,1), Xts(:,2), Yts, 'rb')

hold on

plot_SSVDD(Xtr, Ytr, Xts, alpha, R, rho_epsilon, kernel, param, eta, mycolor);

legend({'','','$\mathcal{S}_\varepsilon$'},'Interpreter','latex','FontSize',10);



