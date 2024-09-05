%% Example 1
clc; clear all; close all;

% Data

mu2 = [-1;-1]; sigma1 = 1;
mu1 = [+1;+1]; sigma2 = 1;

N_points = 1000;

p_A = 0.7; p_O = 0.1;

[Xtr,Ytr] = generate_data(N_points, p_A, p_O, mu1, mu2, sigma1, sigma2);
Ytr = -Ytr;
gscatter(Xtr(:,1), Xtr(:,2), Ytr, 'br')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tau = 0.3;
kernel = 'polynomial';
param = 2;
eta = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = SSVM_Train(Xtr, Ytr, kernel, param, tau, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

b = offset(Xtr, Ytr, alpha, kernel, param, eta, tau);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all

[Xts,Yts] = generate_data(2000, p_A, p_O, mu1, mu2, sigma1, sigma2);
Yts = -Yts;
figure(1)

gscatter(Xts(:,1), Xts(:,2), Yts, 'br')

hold on

p = plot_SSVM(Xtr, Ytr, Xts, alpha, b, kernel, param, eta, [0 1 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xcl,Ycl] = generate_data(5000, p_A, p_O, mu1, mu2, sigma1, sigma2);

ycl_predict = SSVM_Test(Xtr, Ytr, Xcl, alpha, b, kernel, param, eta);

n = size(Xcl,1);
Ycl = -Ycl;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

scores = scoreSVM(Xtr, Ytr, Xcl, Ycl, b, alpha, kernel, param, eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Epsilon = linspace(0.001,0.5,3);

for epsilon = Epsilon
close all
epsilon
qhat = quantile(scores, (ceil((n+1)*(1-epsilon)))/n);
[~,index]=min(abs(qhat-scores));

yhat = Ycl(index);

score_ts_plus = scoreSVM(Xtr, Ytr, Xts, +1, b, alpha, kernel, param, eta);

score_ts_minus = scoreSVM(Xtr, Ytr, Xts, -1, b, alpha, kernel, param, eta);

Cset = [score_ts_plus-qhat <= 0,(score_ts_minus-qhat <= 0)*2];

Cscatter = Cset(:,1) + Cset(:,2);

Xo = [Xts,Cscatter];
Xo = sortrows(Xo,3);

if Xo(1,3) == 0

    figure(1)

    gscatter(Xo(:,1),Xo(:,2),Xo(:,3),'cgry')
    
    legend('void','+1','-1','{+1,-1}')

elseif Xo(1,3) == +1

    figure(1)

    gscatter(Xo(:,1),Xo(:,2),Xo(:,3),'gry')
    
    legend('+1','-1','{+1,-1}')

end
b_eps2 = b  + abs(qhat);

figure(4)

gscatter(Xts(:,1), Xts(:,2), Yts, 'rb')

hold on

plot_SSVM(Xtr, Ytr, Xts, alpha, b_eps2, kernel, param, eta, [0,1,1]);

pause()

end 


