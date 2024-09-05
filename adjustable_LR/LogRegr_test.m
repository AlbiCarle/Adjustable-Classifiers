function y = LogRegr_test(Xtr, Ytr, Xts, alpha, c, c_eps, kernel, param, eta)

K = KernelMatrix(Xtr,Xts,kernel,param);
D = diag(Ytr);

y = sign(1./(1+exp(-(eta*K'*D*alpha - c) + c_eps)) - 0.5); %hai messo un meno davanti a eta