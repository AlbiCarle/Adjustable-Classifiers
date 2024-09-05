%% Example 1
clc; clear all; close all;

% Data

mu1 = [-1;-1]; sigma1 = 1;
mu2 = [+1;+1]; sigma2 = 1;

N_points = 1000;

p_A = 0.7; p_O = 0.00;

[Xtr,Ytr] = generate_data(N_points, p_A, p_O, mu1, mu2, sigma1, sigma2);

%gscatter(Xtr(:,1), Xtr(:,2), Ytr, 'br')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tau = 0.3;
kernel = 'linear';
param = 3;
eta = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = SSVM_Train(Xtr, Ytr, kernel, param, tau, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

b = offset(Xtr, Ytr, alpha, kernel, param, eta, tau);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all

[Xts,Yts] = generate_data(2000, p_A, p_O, mu1, mu2, sigma1, sigma2);

figure(1)

gscatter(Xts(:,1), Xts(:,2), Yts, 'br')

hold on

p = plot_SSVM(Xtr, Ytr, Xts, alpha, b, kernel, param, eta, [0 1 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


