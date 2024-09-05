clc; clear all; close all;

% Data

mu1 = [+1;+1]; sigma1 = 1;
mu2 = [-1;-1]; sigma2 = 1;

N1 = 500;

[Xtr,Ytr] = generate_data(N1, 0.5, 0, mu1, mu2, sigma1, sigma2);

figure(1)

gscatter(Xtr(:,1), Xtr(:,2), Ytr, 'br')

kernel = 'gaussian';
param = 2;
tau = 0.9;
eta = 1;

alpha = LogRegr_train(Xtr, Ytr, kernel, param, tau, eta);

N2 = 1000;

[Xts,Yts] = generate_data(N2, 0.5, 0, mu1, mu2, sigma1, sigma2);


epsilon = 0.1;
delta = 10^(-3);

N_val = ceil(7.47/epsilon*log(1/delta)) + 1; 
                
r = floor(N_val*epsilon*0.5); % scaling
          
[Xvl,Yvl] = generate_data(N_val, 0.5, 0, mu1, mu2, sigma1, sigma2);

c_vector = compute_c(Xtr, Ytr, Xvl, alpha, kernel, param, eta);

[c_sorted, index] = sort(c_vector,'descend');

c_sort = c_vector(index);
c = c_sort(r);

%c = 0; %is always the best choice

y_predict = LogRegr_test(Xtr, Ytr, Xts, alpha, c, kernel, param, eta);

x_true = Xts(y_predict ==1,:);

PSR_pos = sum(y_predict(y_predict==1));

m = [Yts y_predict];

TP = sum((m(:,1)==+1 & m(:,2)==+1));

prob = TP/PSR_pos;

figure(2)

gscatter(Xts(:,1), Xts(:,2), Yts, 'br')

hold on

plot_LogRegr(Xtr, Ytr, Xts, alpha, c, kernel, param, eta, 'g');









