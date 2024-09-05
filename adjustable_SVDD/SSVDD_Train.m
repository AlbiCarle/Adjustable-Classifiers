function alpha = SSVDD_Train(Xtr, Ytr, kernel, param, tau, eta)

n = size(Xtr,1);
if(size(tau,1)>1)
    tau = tau';
end
m = size(tau,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = KernelMatrix(Xtr, Xtr, kernel, param);
D = diag(Ytr);

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% H

H = 2*eta*(D*K*D);

% f

f = D*diag(K);

% unequality constraints

lb = zeros(n,1);

ub = 0.5*((1-2*tau).*Ytr + 1);

% equality constraints

Aeq = Ytr';
beq = 1/(2*eta);
%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options = optimset('Display', 'on');

alpha = quadprog(H,-f,[],[],Aeq,beq,lb,ub,[],options);

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
