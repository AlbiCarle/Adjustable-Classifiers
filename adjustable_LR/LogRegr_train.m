function alpha = LogRegr_train(Xtr, Ytr, kernel, param, tau, eta)

n = size(Xtr,1);

K = KernelMatrix(Xtr, Xtr, kernel, param);
D = diag(Ytr);

C = 0.5*((1-2*tau)*Ytr+1);

L = @(x) 0.5*eta*x'*(D*K*D)*x + x'*log(x) + (C-x)'*log(C-x);

lb = zeros(n,1);
ub = C;

Aeq = Ytr';
beq = 0;

x0 = (lb + ub)/2;

%options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

alpha = fmincon(L,x0,[],[],Aeq,beq,lb,ub, [],[]);







